"""Induced octahedra: bounded discovery, local augmentation and exact lifting.

An octahedron O_d is the complement of d disjoint edges; an induced
O_d is specified by its d antipodal nonedges. The one-for-two search
operates *inside* G; facet amplification constructs an induced
O_(2^(d-1)) in K(G). Neither operation alone certifies divergence.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import networkx as nx

from ._bitsets import iter_indices as _bits
from ._graph_utils import is_simple_graph
from .complete_transversals import CompleteTransversalResult, find_complete_transversal


def _check_graph(graph: nx.Graph) -> None:
    if not is_simple_graph(graph):
        raise ValueError("Require an undirected simple loopless graph")


def _check_pairs(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
) -> tuple[tuple[Hashable, Hashable], ...]:
    _check_graph(graph)
    raw_pairs = tuple(tuple(pair) for pair in antipodal_pairs)
    if len(raw_pairs) < 2 or any(len(pair) != 2 for pair in raw_pairs):
        raise ValueError("Expected at least two antipodal pairs")
    pairs = tuple((a, b) for a, b in raw_pairs)
    vertices = [v for pair in pairs for v in pair]
    if len(set(vertices)) != 2 * len(pairs) or any(v not in graph for v in vertices):
        raise ValueError("Octahedral endpoints must be distinct host vertices")
    for i, (a, b) in enumerate(pairs):
        if graph.has_edge(a, b):
            raise ValueError("Antipodal endpoints must be nonadjacent")
        for u, v in pairs[:i]:
            if not all(graph.has_edge(x, y) for x in (a, b) for y in (u, v)):
                raise ValueError("Antipodal pairs must be completely joined")
    return pairs


def _closed_masks(
    graph: nx.Graph,
) -> tuple[list[Hashable], dict[Hashable, int], list[int]]:
    nodes = list(graph)
    positions = {v: i for i, v in enumerate(nodes)}
    masks = [
        (1 << i) | sum(1 << positions[v] for v in graph[node])
        for i, node in enumerate(nodes)
    ]
    return nodes, positions, masks


def is_dominating_octahedron(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
) -> bool:
    """Whether a valid induced O_d dominates in the *complement* of G.

    Equivalently, no outside vertex is adjacent to every target vertex.
    Raises ValueError if the specified pairs do not induce an octahedron.
    """
    pairs = _check_pairs(graph, antipodal_pairs)
    nodes, positions, masks = _closed_masks(graph)
    common = (1 << len(nodes)) - 1
    for a, b in pairs:
        common &= masks[positions[a]] & masks[positions[b]]
    return common == 0


def find_induced_octahedron(
    graph: nx.Graph,
    dimension: int,
    *,
    max_states: int = 100_000,
    require_dominating: bool = False,
) -> tuple[tuple[Hashable, Hashable], ...] | None:
    """Bounded search for a new induced O_d, without preset vertices.

    Recursive induced-matching search in the complement graph. Returns a
    verified embedding, or None (which can mean exhaustion OR a limit).
    Useful at modest dimensions; it is *not* a practical O24 search
    over all pairs of the 1076-vertex fourth snub iterate.
    """
    _check_graph(graph)
    if type(dimension) is not int or dimension < 2:
        raise ValueError("dimension must be at least 2")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be positive")
    nodes, _, masks = _closed_masks(graph)
    count = 0
    full = (1 << len(nodes)) - 1
    found: tuple[tuple[Hashable, Hashable], ...] | None = None

    def walk(available: int, chosen: list[tuple[int, int]]) -> bool:
        nonlocal count, found
        if len(chosen) == dimension:
            if require_dominating and available:
                return False
            found = tuple((nodes[a], nodes[b]) for a, b in chosen)
            return True
        if available.bit_count() < 2 * (dimension - len(chosen)):
            return False
        for a in _bits(available):
            others = available & ~masks[a] & ~((1 << (a + 1)) - 1)
            for b in _bits(others):
                count += 1
                if count > max_states:
                    return False
                chosen.append((a, b))
                if walk(available & masks[a] & masks[b], chosen):
                    return True
                chosen.pop()
                if count > max_states:
                    return False
        return False

    walk(full, [])
    if found is not None:
        _check_pairs(graph, found)
    return found


@dataclass(frozen=True)
class OctahedralAmplification:
    """Implicit induced octahedron in the clique graph.

    Index s encodes the seed transversal by its binary digits.
    Its opposite has index len(extensions)-1-s. The common
    neighborhoods are unions of all maximal extensions of each
    transversal, not necessarily complete themselves.
    """

    original_pairs: tuple[tuple[Hashable, Hashable], ...]
    extensions: tuple[frozenset[Hashable], ...]
    common_neighborhoods: tuple[frozenset[Hashable], ...]

    @property
    def dimension(self) -> int:
        return len(self.extensions) // 2

    @property
    def antipodal_pairs(
        self,
    ) -> tuple[tuple[frozenset[Hashable], frozenset[Hashable]], ...]:
        return tuple(
            (q, self.extensions[-1 - i])
            for i, q in enumerate(self.extensions[: self.dimension])
        )


def _transversal_mask(pairs: list[tuple[int, int]], index: int) -> int:
    return sum(1 << pair[(index >> i) & 1] for i, pair in enumerate(pairs))


def _common_mask(closed: list[int], vertices: int) -> int:
    common = (1 << len(closed)) - 1
    for v in _bits(vertices):
        common &= closed[v]
    return common


def certify_octahedral_amplification(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    max_facets: int = 65_536,
) -> OctahedralAmplification:
    """Construct maximal extensions and their exact common neighborhoods.

    The seed must have no common closed neighbor. Opposite seed
    transversals have disjoint common closed neighborhoods; nonopposite
    transversals intersect. Hence their distinct maximal extensions
    induce an octahedron of dimension 2**(d-1) in K(G).

    Neither the rest of K(G) nor its edges are constructed. The bound
    and invalid inputs raise ValueError, just as in the original
    amplify_induced_octahedron API.
    """
    pairs = _check_pairs(graph, antipodal_pairs)
    if type(max_facets) is not int or max_facets < 1:
        raise ValueError("max_facets must be a positive integer")
    if len(pairs) >= max_facets.bit_length():
        raise ValueError("Octahedral facet count exceeds max_facets")
    nodes, positions, closed = _closed_masks(graph)
    indexed = [(positions[a], positions[b]) for a, b in pairs]
    endpoints = sum(1 << v for pair in indexed for v in pair)
    if _common_mask(closed, endpoints):
        raise ValueError("A vertex is adjacent to all octahedral vertices")

    extensions: list[frozenset[Hashable]] = []
    neighborhoods: list[frozenset[Hashable]] = []
    for i in range(1 << len(pairs)):
        chosen = _transversal_mask(indexed, i)
        common = _common_mask(closed, chosen)
        neighborhoods.append(frozenset(nodes[v] for v in _bits(common)))
        available = common & ~chosen
        while available:
            bit = available & -available
            chosen |= bit
            available &= closed[bit.bit_length() - 1] & ~bit
        extensions.append(frozenset(nodes[v] for v in _bits(chosen)))
    return OctahedralAmplification(pairs, tuple(extensions), tuple(neighborhoods))


def verify_octahedral_amplification(
    graph: nx.Graph, certificate: OctahedralAmplification
) -> bool:
    """Replay the seed, extensions and opposite disjointness.

    All other intersections follow from the seed transversals. No
    quadratic comparison of the amplified vertices is required.
    """
    try:
        pairs = _check_pairs(graph, certificate.original_pairs)
        nodes, positions, closed = _closed_masks(graph)
        indexed = [(positions[a], positions[b]) for a, b in pairs]
        endpoints = sum(1 << v for pair in indexed for v in pair)
        if _common_mask(closed, endpoints):
            return False
        extensions = certificate.extensions
        size = len(extensions)
        if (
            size != 1 << len(pairs)
            or len(certificate.common_neighborhoods) != size
            or len(set(extensions)) != size
        ):
            return False
        for i, (q, common) in enumerate(
            zip(extensions, certificate.common_neighborhoods, strict=True)
        ):
            if not q <= common or not common <= positions.keys():
                return False
            chosen = _transversal_mask(indexed, i)
            mask = sum(1 << positions[v] for v in q)
            common_mask = sum(1 << positions[v] for v in common)
            if (
                chosen & mask != chosen
                or common_mask != _common_mask(closed, chosen)
                or mask != _common_mask(closed, mask)
                or common & certificate.common_neighborhoods[size - 1 - i]
            ):
                return False
        return True
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def amplified_common_neighbor(
    graph: nx.Graph,
    certificate: OctahedralAmplification,
    *,
    all_extensions: bool = False,
    max_states: int = 1_000_000,
) -> CompleteTransversalResult:
    """Check for a common neighbor without constructing K(G).

    A complete set meeting every selected extension extends to a clique
    meeting all of them. EXHAUSTED proves no common neighbor exists.
    With all_extensions=True, use the union of all possible extensions
    of each seed transversal instead. EXHAUSTED then proves the result
    for every choice of maximal extensions; FOUND in that stronger test
    need not be a common neighbor of the selected family.
    """
    if not verify_octahedral_amplification(graph, certificate):
        raise ValueError("invalid amplification certificate")
    family = (
        certificate.common_neighborhoods if all_extensions else certificate.extensions
    )
    return find_complete_transversal(graph, family, max_states=max_states)


def amplify_induced_octahedron(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    max_facets: int = 1024,
) -> tuple[tuple[frozenset[Hashable], frozenset[Hashable]], ...]:
    """Realize O_(2^(d-1)) in K(G) from a complement-dominating O_d.

    Backward-compatible wrapper returning only the antipodal pairs.
    Does not construct K(G) or claim indefinite amplification.
    Raises ValueError on invalid inputs or exceeding max_facets.
    """
    return certify_octahedral_amplification(
        graph, antipodal_pairs, max_facets=max_facets
    ).antipodal_pairs


def augment_induced_octahedron(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    max_walk_steps: int = 1000,
    max_checks: int = 200_000,
    seed: int = 9883,
    require_dominating: bool = False,
) -> tuple[tuple[Hashable, Hashable], ...] | None:
    """Heuristically grow an induced O_d into an induced O_(d+1).

    One-for-two augmentation replaces one antipodal pair by two pairs
    adjacent to all the retained vertices. When no immediate augmentation
    exists, randomly replace one pair and retry, within explicit budgets.
    Set require_dominating=True to demand that the *new* octahedron
    have no external common neighbor. Returns a validated pair list or
    None (INCONCLUSIVE, never proof that no larger octahedron exists).
    No graph-specific vertices are hardcoded.
    """
    pairs = _check_pairs(graph, antipodal_pairs)
    if type(max_walk_steps) is not int or max_walk_steps < 0:
        raise ValueError("max_walk_steps must be nonnegative")
    if type(max_checks) is not int or max_checks < 1:
        raise ValueError("max_checks must be positive")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    nodes, positions, masks = _closed_masks(graph)
    work = [(positions[a], positions[b]) for a, b in pairs]
    full = (1 << len(nodes)) - 1
    rng = random.Random(seed)
    checks = 0

    def common(selected: list[tuple[int, int]]) -> int:
        remaining = full
        for u, v in selected:
            remaining &= masks[u] & masks[v]
        return remaining

    def charge() -> bool:
        nonlocal checks
        checks += 1
        return checks <= max_checks

    for step in range(max_walk_steps + 1):
        for omitted in range(len(work)):
            rest = work[:omitted] + work[omitted + 1 :]
            candidate_mask = common(rest)
            for a in _bits(candidate_mask):
                others = candidate_mask & ~masks[a] & ~((1 << (a + 1)) - 1)
                for b in _bits(others):
                    if not charge():
                        return None
                    remaining = candidate_mask & masks[a] & masks[b]
                    for c in _bits(remaining):
                        others2 = remaining & ~masks[c] & ~((1 << (c + 1)) - 1)
                        for d in _bits(others2):
                            if not charge():
                                return None
                            grown = rest + [(a, b), (c, d)]
                            if not require_dominating or common(grown) == 0:
                                answer = tuple((nodes[u], nodes[v]) for u, v in grown)
                                _check_pairs(graph, answer)
                                return answer

        if step == max_walk_steps:
            break
        omitted = rng.randrange(len(work))
        candidate_mask = common(work[:omitted] + work[omitted + 1 :])
        picked: tuple[int, int] | None = None
        seen = 0
        for a in _bits(candidate_mask):
            others = candidate_mask & ~masks[a] & ~((1 << (a + 1)) - 1)
            for b in _bits(others):
                if not charge():
                    return None
                seen += 1
                if rng.randrange(seen) == 0:
                    picked = (a, b)
        if picked is not None:
            work[omitted] = picked
    return None
