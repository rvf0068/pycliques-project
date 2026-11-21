"""Induced octahedra: bounded discovery, local augmentation and exact lifting.

An octahedron O_d is the complement of d disjoint edges; an induced
O_d is specified by its d antipodal nonedges. The one-for-two search
operates *inside* G; facet amplification constructs an induced
O_(2^(d-1)) in K(G). Neither operation alone certifies divergence.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence

import networkx as nx


def _check_graph(graph: nx.Graph) -> None:
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
    ):
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


def _bits(mask: int):
    while mask:
        one = mask & -mask
        yield one.bit_length() - 1
        mask ^= one


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


def amplify_induced_octahedron(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    max_facets: int = 1024,
) -> tuple[tuple[frozenset[Hashable], frozenset[Hashable]], ...]:
    """Realize O_(2^(d-1)) in K(G) from a complement-dominating O_d in G.

    Vertices of K(G) are maximal completes of G, returned as frozensets.
    Opposite pairs of extended facets become the antipodal pairs of
    the lifted induced octahedron. Does not construct the full K(G).

    Raises ValueError if the target is invalid, has an external common
    neighbor, or its 2^d facets exceed max_facets. This is a one-step
    construction, not a claim of automatic infinite amplification.
    """
    pairs = _check_pairs(graph, antipodal_pairs)
    if type(max_facets) is not int or max_facets < 1:
        raise ValueError("max_facets must be a positive integer")
    d = len(pairs)
    if 1 << d > max_facets:
        raise ValueError("Octahedral facet count exceeds max_facets")
    if not is_dominating_octahedron(graph, pairs):
        raise ValueError("A vertex is adjacent to all octahedral vertices")

    facets: list[frozenset[Hashable]] = []
    for mask in range(1 << d):
        complete: set[Hashable] = {
            pair[(mask >> i) & 1] for i, pair in enumerate(pairs)
        }
        for candidate in graph:
            if candidate not in complete and all(
                graph.has_edge(candidate, v) for v in complete
            ):
                complete.add(candidate)
        facets.append(frozenset(complete))
    full = (1 << d) - 1
    return tuple(
        (facets[mask], facets[mask ^ full])
        for mask in range(1 << d)
        if mask < (mask ^ full)
    )


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
