"""Replayable homotopy-preserving edge reductions and vertex-star splitting.

Cone-link edge deletions and dominated-vertex folds use bitset adjacency
for dense input graphs. A small residual core may also undergo edge
deletions whose common-neighborhood clique complexes have an exact
contractible Dong certificate. Every move has an independent replayer.

The vertex-star recognizer uses exact augmented Dong critical cells in
the deletion and link. It accepts only justified CW models and a
nullhomotopic inclusion established by a connectivity/dimension gap.
Expensive searches are opt-in, not part of the default census classifier.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import networkx as nx

from .simplex import _stable_vertex_key
from .symbolic_dong import symbolic_dong_certificate


@dataclass(frozen=True)
class FlagMove:
    """A step: cone_edge(u,v,apex), fold_vertex(u,v), or contractible_edge(u,v)."""

    kind: str
    u: Hashable
    v: Hashable
    apex: Hashable | None = None


@dataclass(frozen=True)
class FlagReductionCertificate:
    """All local proofs, followed by the surviving labelled graph."""

    moves: tuple[FlagMove, ...]
    remaining_vertices: tuple[Hashable, ...]
    remaining_edges: tuple[tuple[Hashable, Hashable], ...]


@dataclass(frozen=True)
class VertexStarWedgeCertificate:
    """A certified nullhomotopic link inclusion and resulting wedge."""

    vertex: Hashable
    deletion_critical: dict[int, int]
    link_critical: dict[int, int]
    deletion_spheres: dict[int, int]
    link_spheres: dict[int, int]
    spheres: dict[int, int]


@dataclass(frozen=True)
class ReducedStarWedgeCertificate:
    reduction: FlagReductionCertificate
    star: VertexStarWedgeCertificate
    spheres: dict[int, int]


def _canonical_core(graph: nx.Graph) -> nx.Graph:
    """Reindex an arbitrary hashable-labelled core reproducibly."""
    if all(isinstance(v, int) for v in graph):
        labels = sorted(graph)
    else:
        labels = sorted(graph, key=_stable_vertex_key)
    return nx.relabel_nodes(
        graph, {v: i for i, v in enumerate(labels)}, copy=True
    )


def _valid_graph(graph: nx.Graph) -> bool:
    return not (
        graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph)
    )


def _indices(mask: int):
    while mask:
        bit = mask & -mask
        mask ^= bit
        yield bit.bit_length() - 1


def _cone_apex(adj: list[int], u: int, v: int) -> int | None:
    common = adj[u] & adj[v]
    remaining = common
    while remaining:
        bit = remaining & -remaining
        remaining ^= bit
        w = bit.bit_length() - 1
        if not common & ~(adj[w] | bit):
            return w
    return None


def _cone_fold_passes(
    graph: nx.Graph, *, passes: int, max_moves: int, start_round: int = 1
) -> tuple[nx.Graph, list[FlagMove]]:
    labels = tuple(graph)
    n = len(labels)
    index = {v: i for i, v in enumerate(labels)}
    adj = [sum(1 << index[v] for v in graph[u]) for u in labels]
    alive = [True] * n
    moves: list[FlagMove] = []
    for stage in range(start_round, start_round + passes):
        edges = [
            (u, v)
            for u in range(n)
            if alive[u]
            for v in range(u + 1, n)
            if adj[u] >> v & 1
        ]
        if stage % 2 == 0:
            random.Random(stage).shuffle(edges)
        changed = 0
        for u, v in edges:
            if not adj[u] >> v & 1:
                continue
            apex = _cone_apex(adj, u, v)
            if apex is None:
                continue
            if len(moves) >= max_moves:
                raise OverflowError("edge-reduction move limit")
            moves.append(FlagMove("cone_edge", labels[u], labels[v], labels[apex]))
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            changed += 1
        for u in range(n):
            if not alive[u]:
                continue
            for v in _indices(adj[u]):
                if not adj[u] & ~(adj[v] | (1 << v)):
                    if len(moves) >= max_moves:
                        raise OverflowError("edge-reduction move limit")
                    moves.append(FlagMove("fold_vertex", labels[u], labels[v]))
                    for w in _indices(adj[u]):
                        adj[w] ^= 1 << u
                    adj[u] = 0
                    alive[u] = False
                    changed += 1
                    break
        if not changed:
            break
    result = nx.Graph()
    result.add_nodes_from(labels[i] for i in range(n) if alive[i])
    result.add_edges_from(
        (labels[u], labels[v])
        for u in range(n)
        if alive[u]
        for v in range(u + 1, n)
        if adj[u] >> v & 1
    )
    return result, moves


def _contractible_edge_passes(
    graph: nx.Graph,
    *,
    passes: int,
    max_moves: int,
    max_link_vertices: int,
    max_nodes: int,
) -> tuple[nx.Graph, list[FlagMove]]:
    current = graph.copy()
    moves: list[FlagMove] = []
    for stage in range(passes):
        edges = list(current.edges)
        random.Random(stage + 14).shuffle(edges)
        changed = 0
        for u, v in edges:
            if not current.has_edge(u, v):
                continue
            common = set(current[u]) & set(current[v])
            if not common or len(common) > max_link_vertices:
                continue
            proof = symbolic_dong_certificate(
                current.subgraph(common),
                max_vertices=max_link_vertices,
                max_nodes=max_nodes,
            )
            if proof is None or proof.critical:
                continue
            if len(moves) >= max_moves:
                raise OverflowError("contractible-link move limit")
            moves.append(FlagMove("contractible_edge", u, v))
            current.remove_edge(u, v)
            changed += 1
        if not changed:
            break
    return current, moves


def reduce_flag_graph(
    graph: nx.Graph,
    *,
    cone_passes: int = 9,
    contractible_passes: int = 5,
    max_moves: int = 250_000,
    max_contractible_vertices: int = 80,
    max_vertices: int = 1000,
    max_link_vertices: int = 48,
    max_nodes: int = 100_000,
) -> tuple[nx.Graph, FlagReductionCertificate] | None:
    """Find and replay a bounded homotopy-preserving graph reduction.

    Each cone edge has a universal vertex in its nonempty common-neighbor
    graph. Each fold has N[u] contained in N[v]. A contractible edge has a
    nonempty common-neighbor flag complex with zero augmented Dong critical
    faces. All three moves preserve the clique complex's homotopy type.
    A resource refusal is inconclusive, not a proof.
    """
    if (
        not _valid_graph(graph)
        or len(graph) > max_vertices
        or cone_passes < 1
        or contractible_passes < 0
        or max_contractible_vertices < 0
        or min(max_moves, max_nodes, max_link_vertices) < 1
    ):
        return None
    try:
        current, moves = _cone_fold_passes(
            graph, passes=cone_passes - 1, max_moves=max_moves
        )
        tail, extra = _cone_fold_passes(
            current, passes=1, max_moves=max_moves - len(moves)
        )
        moves.extend(extra)
        current = tail
        if contractible_passes and len(current) <= max_contractible_vertices:
            current, extra = _contractible_edge_passes(
                current,
                passes=contractible_passes,
                max_moves=max_moves - len(moves),
                max_link_vertices=max_link_vertices,
                max_nodes=max_nodes,
            )
            moves.extend(extra)
    except (MemoryError, RecursionError, OverflowError):
        return None
    certificate = FlagReductionCertificate(
        tuple(moves), tuple(current), tuple(current.edges)
    )
    return (
        (current, certificate)
        if verify_flag_reduction(graph, certificate, max_nodes=max_nodes)
        else None
    )


def verify_flag_reduction(
    graph: nx.Graph,
    certificate: FlagReductionCertificate,
    *,
    max_nodes: int = 250_000,
) -> bool:
    """Check every move against the *current* graph, not saved search state."""
    if not _valid_graph(graph):
        return False
    try:
        current = graph.copy()
        for move in certificate.moves:
            u, v = move.u, move.v
            if not current.has_edge(u, v):
                return False
            if move.kind == "cone_edge":
                common = set(current[u]) & set(current[v])
                apex = move.apex
                if apex not in common or not common - {apex} <= set(current[apex]):
                    return False
                current.remove_edge(u, v)
            elif move.kind == "fold_vertex":
                if not set(current[u]) | {u} <= set(current[v]) | {v}:
                    return False
                current.remove_node(u)
            elif move.kind == "contractible_edge":
                common = set(current[u]) & set(current[v])
                if not common:
                    return False
                proof = symbolic_dong_certificate(
                    current.subgraph(common), max_vertices=80, max_nodes=max_nodes
                )
                if proof is None or proof.critical:
                    return False
                current.remove_edge(u, v)
            else:
                return False
        expected_edges = {frozenset(e) for e in certificate.remaining_edges}
        return (
            set(current) == set(certificate.remaining_vertices)
            and {frozenset(e) for e in current.edges} == expected_edges
        )
    except (KeyError, TypeError, ValueError, MemoryError, RecursionError):
        return False


def _dong_wedge(
    graph: nx.Graph,
) -> tuple[dict[int, int], dict[int, int]] | None:
    """Get exact wedge data for special connected augmented Morse CW models."""
    if not graph or not nx.is_connected(graph):
        return None
    certificate = symbolic_dong_certificate(graph, max_vertices=80)
    if certificate is None:
        return None
    critical = certificate.critical
    if -1 in critical:
        return None
    if not critical:
        return critical, {}
    if len(critical) == 1 and 0 not in critical:
        return critical, dict(critical)
    # Restoring the vertex matched with the augmented empty face gives
    # two 0-cells and one 1-cell: connectedness forces that edge to join
    # the vertices (a tree). All higher cells attach to a contractible base.
    if critical.get(0) == critical.get(1) == 1 and len(critical) <= 3:
        return critical, {d: count for d, count in critical.items() if d > 1}
    # The remaining 1-skeleton is a wedge of circles. Since pi_k of this
    # space vanishes for k >= 2, d>=3 cells attach trivially.
    if len(critical) == 2 and 1 in critical and all(d == 1 or d >= 3 for d in critical):
        return critical, dict(critical)
    return None


def vertex_star_wedge_certificate(
    graph: nx.Graph,
    *,
    candidates: Iterable[Hashable] | None = None,
    max_candidates: int = 80,
) -> VertexStarWedgeCertificate | None:
    """Certify a vertex-star cofiber splitting with a null link map.

    The deletion and link must be connected certified wedges. If the base
    has a positive sphere, its minimum dimension must be at least two
    and strictly exceed every link sphere dimension. The inclusion is then
    nullhomotopic because the base is sufficiently connected.
    """
    if not _valid_graph(graph) or len(graph) > 80 or max_candidates < 1:
        return None
    vertices = tuple(graph) if candidates is None else tuple(candidates)
    for vertex in vertices[:max_candidates]:
        if vertex not in graph:
            continue
        deletion = graph.copy()
        deletion.remove_node(vertex)
        if not deletion or not nx.is_connected(deletion):
            continue
        link = graph.subgraph(graph[vertex]).copy()
        deletion_data = _dong_wedge(deletion)
        link_data = _dong_wedge(link)
        if deletion_data is None or link_data is None:
            continue
        base_critical, base = deletion_data
        link_critical, source = link_data
        if base and (min(base) < 2 or (source and max(source) >= min(base))):
            continue
        spheres = dict(Counter(base) + Counter({d + 1: c for d, c in source.items()}))
        certificate = VertexStarWedgeCertificate(
            vertex, base_critical, link_critical, base, source, spheres
        )
        if verify_vertex_star_wedge(graph, certificate):
            return certificate
    return None


def verify_vertex_star_wedge(
    graph: nx.Graph, certificate: VertexStarWedgeCertificate
) -> bool:
    """Recompute both Morse models and verify the dimension-gap theorem."""
    if not _valid_graph(graph) or certificate.vertex not in graph:
        return False
    deletion = graph.copy()
    deletion.remove_node(certificate.vertex)
    if not deletion or not nx.is_connected(deletion):
        return False
    link = graph.subgraph(graph[certificate.vertex]).copy()
    deletion_data = _dong_wedge(deletion)
    link_data = _dong_wedge(link)
    if deletion_data is None or link_data is None:
        return False
    critical_a, base = deletion_data
    critical_b, source = link_data
    if base and (min(base) < 2 or (source and max(source) >= min(base))):
        return False
    expected = dict(Counter(base) + Counter({d + 1: c for d, c in source.items()}))
    return (
        critical_a == certificate.deletion_critical
        and critical_b == certificate.link_critical
        and base == certificate.deletion_spheres
        and source == certificate.link_spheres
        and expected == certificate.spheres
    )


def reduced_star_wedge_certificate(
    graph: nx.Graph,
    *,
    max_vertices: int = 1000,
    candidates: Iterable[int] | None = None,
    **reduction_options,
) -> ReducedStarWedgeCertificate | None:
    """Try dense edge reductions and a vertex-star split; opt-in and bounded.

    candidates are vertex indices *after* sorting/relabeling the reduced core.
    Graph labels are preserved in the reduction certificate.
    """
    if not graph or len(graph) > max_vertices:
        return None
    result = reduce_flag_graph(graph, **reduction_options)
    if result is None:
        return None
    core, reduction = result
    core = _canonical_core(core)
    star = vertex_star_wedge_certificate(core, candidates=candidates)
    if star is None:
        return None
    proof = ReducedStarWedgeCertificate(reduction, star, star.spheres)
    return proof if verify_reduced_star_wedge(graph, proof) else None


def verify_reduced_star_wedge(
    graph: nx.Graph, certificate: ReducedStarWedgeCertificate
) -> bool:
    """Independently replay the full graph reduction and vertex-star proof."""
    if not verify_flag_reduction(graph, certificate.reduction):
        return False
    core = nx.Graph()
    core.add_nodes_from(certificate.reduction.remaining_vertices)
    core.add_edges_from(certificate.reduction.remaining_edges)
    core = _canonical_core(core)
    return (
        verify_vertex_star_wedge(core, certificate.star)
        and certificate.spheres == certificate.star.spheres
    )
