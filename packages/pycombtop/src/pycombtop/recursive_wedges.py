"""Replayable vertex-star decompositions of finite flag complexes.

Restrict the link inclusion to a dominated-vertex retract of its source
and compose it with a dominated-vertex retraction of the deletion. If its
image lies in a dismantlable induced subgraph, the original map is null.
Alternatively cellular approximation gives nullity when every sphere in
an exact link model lies below the first positive cell of the connected
base. In either case the cofiber is the base wedged with the suspension
of the link (Hatcher, Algebraic Topology, Proposition 0.18).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import combinations

import networkx as nx

from .attachment_certificates import _replay_fold_map
from .s_collapses import verify_dismantling_certificate
from .wedge_certificates import certified_flag_wedge

Moves = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ImageCarrier:
    link_folds: Moves
    deletion_folds: Moves
    extra: tuple[int, ...]
    contraction: Moves


@dataclass(frozen=True)
class WedgeStep:
    rule: str
    children: tuple[int, ...] = ()
    moves: Moves = ()
    vertex: int = -1
    carrier: ImageCarrier | None = None


@dataclass(frozen=True)
class RecursiveWedgeCertificate:
    # The i-th label is represented by bit 1 << i in the subproblem keys.
    labels: tuple[Hashable, ...]
    steps: dict[int, WedgeStep]
    spheres: dict[int, int]


def _indices(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _join_types(factors: list[dict[int, int]]) -> dict[int, int]:
    result = factors[0]
    for factor in factors[1:]:
        joined: Counter[int] = Counter()
        for d, a in result.items():
            for e, b in factor.items():
                joined[d + e + 1] += a * b
        result = dict(joined)
    return result


def _has_gap(base: dict[int, int], link: dict[int, int]) -> bool:
    return not base or not link or max(link) < min(base)


def _cofiber(base: dict[int, int], link: dict[int, int]) -> dict[int, int]:
    result = Counter(base)
    result.update({d + 1: n for d, n in link.items()})
    return dict(result)


def _check_carrier(deletion: nx.Graph, link: nx.Graph, proof: ImageCarrier) -> bool:
    # Track every original link vertex through both retractions.
    source = _replay_fold_map(link, tuple(link), proof.link_folds)
    if source is None:
        return False
    _, link_image = source
    target = _replay_fold_map(deletion, link_image, proof.deletion_folds)
    if target is None:
        return False
    core, image = target
    vertices = set(image).union(proof.extra)
    return vertices <= core.nodes and verify_dismantling_certificate(
        core.subgraph(vertices), proof.contraction
    )


def verify_recursive_wedge(
    graph: nx.Graph, certificate: RecursiveWedgeCertificate
) -> bool:
    """Reconstruct each induced graph and evaluate the supplied proof bottom-up."""
    try:
        if (
            not graph
            or graph.is_directed()
            or graph.is_multigraph()
            or nx.number_of_selfloops(graph)
        ):
            return False
        labels = certificate.labels
        if len(labels) != len(graph) or set(labels) != set(graph):
            return False
        indexed = nx.relabel_nodes(
            graph, {v: i for i, v in enumerate(labels)}, copy=True
        )
        full = (1 << len(labels)) - 1
        types: dict[int, dict[int, int]] = {}
        for mask in sorted(certificate.steps, key=int.bit_count):
            if mask <= 0 or mask & ~full:
                return False
            subgraph = indexed.subgraph(_indices(mask))
            step = certificate.steps[mask]
            if any(c not in types or c & ~mask for c in step.children):
                return False
            if step.rule == "leaf":
                if step.children:
                    return False
                result = certified_flag_wedge(subgraph)
                if result is None:
                    return False
            elif step.rule == "fold":
                reduced = _replay_fold_map(subgraph, (), step.moves)
                if reduced is None or step.children != (
                    sum(1 << i for i in reduced[0]),
                ):
                    return False
                result = types[step.children[0]]
            elif step.rule == "join":
                parts = nx.connected_components(nx.complement(subgraph))
                expected = sorted(sum(1 << i for i in part) for part in parts)
                if len(expected) < 2 or sorted(step.children) != expected:
                    return False
                result = _join_types([types[c] for c in step.children])
            elif step.rule == "split":
                v = step.vertex
                if v not in subgraph:
                    return False
                link = subgraph.subgraph(subgraph[v])
                deletion = subgraph.subgraph(set(subgraph) - {v})
                if not link or not deletion or not nx.is_connected(deletion):
                    return False
                if step.children != (mask ^ (1 << v), sum(1 << i for i in link)):
                    return False
                base, source = (types[c] for c in step.children)
                if step.carrier is None:
                    if not _has_gap(base, source):
                        return False
                elif not _check_carrier(deletion, link, step.carrier):
                    return False
                result = _cofiber(base, source)
            else:
                return False
            types[mask] = result
        return full in types and types[full] == certificate.spheres
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return False


def recursive_wedge_certificate(
    graph: nx.Graph,
    *,
    max_vertices: int = 40,
    max_states: int = 640,
) -> RecursiveWedgeCertificate | None:
    """Search induced subgraphs with a shared state limit, then replay the proof.

    Disconnected leaves are accepted only for discrete spaces. Splittings
    require a connected deletion; dimension zero in a discrete link gives
    the additional circles in its unreduced suspension.
    """
    if not 0 < len(graph) <= max_vertices or max_states < 1:
        return None
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        return None
    labels = tuple(graph)
    index = {v: i for i, v in enumerate(labels)}
    adj = [sum(1 << index[w] for w in graph[v]) for v in labels]
    full = (1 << len(labels)) - 1
    steps: dict[int, WedgeStep] = {}
    states = 0

    @lru_cache(maxsize=2048)
    def folds(mask: int) -> tuple[int, Moves]:
        moves: list[tuple[int, int]] = []
        while mask.bit_count() > 1:
            for v in _indices(mask):
                neighbors = mask & adj[v]
                w = next(
                    (
                        w
                        for w in _indices(neighbors)
                        if not neighbors & ~(adj[w] | (1 << w))
                    ),
                    None,
                )
                if w is not None:
                    moves.append((v, w))
                    mask ^= 1 << v
                    break
            else:
                break
        return mask, tuple(moves)

    def components(mask: int, *, complement: bool = False) -> list[int]:
        remaining = mask
        parts = []
        while remaining:
            frontier = remaining & -remaining
            part = 0
            while frontier:
                part |= frontier
                remaining ^= frontier
                adjacent = 0
                for i in _indices(frontier):
                    adjacent |= ~(adj[i] | (1 << i)) if complement else adj[i]
                frontier = remaining & adjacent
            parts.append(part)
        return parts

    def carrier(deletion: int, link: int) -> ImageCarrier | None:
        image, source_moves = folds(link)
        target, target_moves = folds(deletion)
        for a, b in target_moves:
            if image & (1 << a):
                image = (image ^ (1 << a)) | (1 << b)
        outside = tuple(_indices(target ^ image))
        for size in range(3):
            for extra in combinations(outside, size):
                survivor, contraction = folds(image | sum(1 << i for i in extra))
                if survivor.bit_count() == 1:
                    return ImageCarrier(source_moves, target_moves, extra, contraction)
        return None

    class StateLimit(Exception):
        pass

    @cache
    def solve(mask: int) -> dict[int, int] | None:
        nonlocal states
        states += 1
        if states > max_states:
            raise StateLimit
        core, moves = folds(mask)
        if core != mask:
            result = solve(core)
            if result is not None:
                steps[mask] = WedgeStep("fold", (core,), moves)
            return result
        edges = sum((adj[i] & mask).bit_count() for i in _indices(mask)) // 2
        if not edges:
            steps[mask] = WedgeStep("leaf")
            return {0: mask.bit_count() - 1} if mask.bit_count() > 1 else {}
        parts = components(mask, complement=True)
        if len(parts) > 1:
            factors = []
            for part in parts:
                factor = solve(part)
                if factor is None:
                    return None
                factors.append(factor)
            steps[mask] = WedgeStep("join", tuple(parts))
            return _join_types(factors)
        if len(components(mask)) != 1:
            return None
        if not any(
            adj[i] & adj[j] & mask
            for i in _indices(mask)
            for j in _indices(adj[i] & mask)
        ):
            steps[mask] = WedgeStep("leaf")
            rank = edges - mask.bit_count() + 1
            return {1: rank} if rank else {}

        candidates = []
        for v in _indices(mask):
            deletion = mask ^ (1 << v)
            link = mask & adj[v]
            if len(components(deletion)) != 1:
                continue
            size = folds(deletion)[0].bit_count() + folds(link)[0].bit_count()
            candidates.append((size, link.bit_count(), v, deletion, link))
        for _, _, v, deletion, link in sorted(candidates):
            base = solve(deletion)
            if base is None:
                continue
            source = solve(link)
            if source is None:
                continue
            witness = None
            if not _has_gap(base, source):
                witness = carrier(deletion, link)
                if witness is None:
                    continue
            steps[mask] = WedgeStep(
                "split", (deletion, link), vertex=v, carrier=witness
            )
            return _cofiber(base, source)
        return None

    try:
        spheres = solve(full)
    except StateLimit:
        return None
    finally:
        solve.cache_clear()
        folds.cache_clear()
    if spheres is None:
        return None
    # Retain only the subproblems used in the successful decomposition.
    proof = {}
    todo = [full]
    while todo:
        mask = todo.pop()
        if mask not in proof:
            proof[mask] = steps[mask]
            todo.extend(steps[mask].children)
    certificate = RecursiveWedgeCertificate(labels, proof, spheres)
    return certificate if verify_recursive_wedge(graph, certificate) else None
