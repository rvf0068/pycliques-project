"""Independently test forced-octahedron pruning against exhaustive oracles."""

from __future__ import annotations

import random
from itertools import combinations

import networkx as nx
from pycliques.octahedral_blind import (
    OctahedralBlindStatus,
    complete_dominating_octahedron,
    screen_partial_octahedral_retraction,
)
from pycliques.octahedral_forced_pairs import screen_forced_pairs


def _nonedge(graph: nx.Graph) -> tuple[int, ...]:
    full = (1 << len(graph)) - 1
    return tuple(full & ~(1 << x) & ~sum(1 << y for y in graph[x]) for x in graph)


def _oracle(nonedge: tuple[int, ...], common: int, cache: dict[int, bool]) -> bool:
    """Exact induced-matching domination oracle on a small vertex set."""
    if common == 0:
        return True
    if common in cache:
        return cache[common]
    vertices = [x for x in range(len(nonedge)) if common & (1 << x)]
    edges = [(u, v) for u, v in combinations(vertices, 2) if nonedge[u] & (1 << v)]
    options = {
        x: [(u, v) for u, v in edges if nonedge[x] & ((1 << u) | (1 << v))]
        for x in vertices
    }
    pivot = min(options, key=lambda x: len(options[x]))
    answer = any(
        _oracle(nonedge, common & ~(nonedge[u] | nonedge[v]), cache)
        for u, v in options[pivot]
    )
    cache[common] = answer
    return answer


def test_randomized_against_independent_exact_oracle() -> None:
    rng = random.Random(20261123)
    rejected = 0
    for _ in range(400):
        n = rng.randint(5, 10)
        probability = rng.uniform(0.2, 0.8)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(
            (u, v) for u, v in combinations(range(n), 2) if rng.random() < probability
        )
        nonedge = _nonedge(graph)
        common = rng.randrange(1, 1 << n)
        result = screen_forced_pairs(nonedge, common)
        if not result.possible:
            rejected += 1
            assert not _oracle(nonedge, common, {})
    assert rejected > 50


def test_planted_retractions_survive_all_prefixes() -> None:
    rng = random.Random(260040)
    for _ in range(90):
        d = rng.randint(3, 6)
        n = 2 * d + rng.randint(0, 6)
        graph = nx.complete_graph(n)
        images = list(range(2 * d)) + [rng.randrange(2 * d) for _ in range(n - 2 * d)]
        for u, v in combinations(range(n), 2):
            if images[u] == (images[v] ^ 1) or (
                (u >= 2 * d or v >= 2 * d) and rng.random() < 0.3
            ):
                graph.remove_edge(u, v)
        nonedge = _nonedge(graph)
        for prefix in range(d + 1):
            common = (1 << n) - 1
            for j in range(prefix):
                common &= ~(nonedge[2 * j] | nonedge[2 * j + 1])
            assert screen_forced_pairs(nonedge, common).possible


def test_second_round_isolated_vertex_and_forced_degree() -> None:
    isolated = nx.from_graph6_bytes(b"H~{^{A^")
    result = screen_forced_pairs(_nonedge(isolated), (1 << 9) - 1)
    assert not result.possible
    assert result.reason == "ISOLATED"
    assert result.forced_pairs == ((5, 7),)
    assert result.rounds == 2
    assert result.witness == (8,)

    degree = nx.from_graph6_bytes(b"K^n~pmvzKm~r")
    result = screen_forced_pairs(_nonedge(degree), (1 << 12) - 1)
    assert not result.possible
    assert result.reason == "FORCED_DEGREE"
    assert result.forced_pairs == ((6, 9),)
    assert result.rounds == 2


def test_partial_target_integration_and_positive_completion() -> None:
    # The first two vertices induce O1 in X and are completely joined to
    # the nine-vertex obstruction. The partial-target screen must reject
    # any further octahedral extension of this selected antipodal pair.
    base = nx.from_graph6_bytes(b"H~{^{A^")
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from((u + 2, v + 2) for u, v in base.edges())
    graph.add_edges_from((a, x) for a in (0, 1) for x in range(2, 11))
    screen = screen_partial_octahedral_retraction(graph, [(0, 1)])
    assert not screen.possible
    assert screen.reason == "FORCED_PAIR_ISOLATED"

    octahedron = nx.complete_multipartite_graph(2, 2, 2, 2)
    completion = complete_dominating_octahedron(
        octahedron,
        [(0, 1), (2, 3)],
        target_dimension=4,
        max_states=100,
    )
    assert completion.status is OctahedralBlindStatus.FOUND
    assert set(completion.antipodal_pairs or ()) == {(0, 1), (2, 3), (4, 5), (6, 7)}


def test_forced_pair_recursion_propagates_exhaustion_and_limit() -> None:
    # In H = K2 + K2 + C5, after fixing the first K2, the next K2
    # is compulsory. However, a single antipodal pair cannot dominate
    # the remaining C5 by open H-neighborhoods. This exercises the
    # forced-recursion negative result (not just an immediate obstruction).
    complement = nx.disjoint_union_all(
        [nx.complete_graph(2), nx.complete_graph(2), nx.cycle_graph(5)]
    )
    graph = nx.complement(complement)
    check = screen_forced_pairs(_nonedge(graph), ((1 << len(graph)) - 1) & ~3)
    assert check.possible
    assert check.forced_pairs == ((2, 3),)

    result = complete_dominating_octahedron(
        graph, [(0, 1)], target_dimension=2, max_states=100
    )
    assert result.status is OctahedralBlindStatus.EXHAUSTED

    limited = complete_dominating_octahedron(
        graph, [(0, 1)], target_dimension=2, max_states=1
    )
    assert limited.status is OctahedralBlindStatus.LIMIT_REACHED


def test_forced_pairs_exceed_requested_target_dimension() -> None:
    # An O4 host has four compulsory antipodal pairs; retaining one
    # while requesting O2 leaves insufficient dimensions for the other
    # three. This must be an exact EXHAUSTED result.
    graph = nx.complete_multipartite_graph(2, 2, 2, 2)
    result = complete_dominating_octahedron(
        graph, [(0, 1)], target_dimension=2, max_states=100
    )
    assert result.status is OctahedralBlindStatus.EXHAUSTED


def test_no_forced_pair_fallback_finds_and_exhausts() -> None:
    """The ordinary completion search must remain complete without forcing.

    In the complement, C5 has no induced matching of two edges, C6 and
    C7 admit dominating induced matchings of two edges, C8 cannot have
    three, and C9 does admit three. All cycles have minimum degree two;
    neither isolated-vertex nor compulsory-pair propagation settles the
    initial residual.
    """
    for length, target_dimension, expected in (
        (5, 3, OctahedralBlindStatus.EXHAUSTED),
        (6, 3, OctahedralBlindStatus.FOUND),
        (7, 3, OctahedralBlindStatus.FOUND),
        (8, 4, OctahedralBlindStatus.EXHAUSTED),
        (9, 4, OctahedralBlindStatus.FOUND),
    ):
        complement = nx.disjoint_union(nx.complete_graph(2), nx.cycle_graph(length))
        graph = nx.complement(complement)
        common = ((1 << len(graph)) - 1) & ~3
        screen = screen_forced_pairs(_nonedge(graph), common)
        assert screen.possible and screen.forced_pairs == ()
        result = complete_dominating_octahedron(
            graph, [(0, 1)], target_dimension=target_dimension, max_states=10000
        )
        assert result.status is expected
        if result.status is OctahedralBlindStatus.FOUND:
            assert result.states_explored > 1
            assert result.antipodal_pairs is not None
            assert len(result.antipodal_pairs) == target_dimension

    # A target dimension too large for C6 is excluded by the ordinary
    # exact search, rather than by the forced-pair screen.
    complement = nx.disjoint_union(nx.complete_graph(2), nx.cycle_graph(6))
    graph = nx.complement(complement)
    overlarge = complete_dominating_octahedron(
        graph, [(0, 1)], target_dimension=4, max_states=10000
    )
    assert overlarge.status is OctahedralBlindStatus.EXHAUSTED
