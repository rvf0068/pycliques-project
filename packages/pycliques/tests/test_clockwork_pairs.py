"""Tests for the Theorem 2.6 / Theorem 3.1 clockwork pair map machinery."""

import networkx as nx
import pytest
from pycliques.clockwork_pairs import (
    candidate_target_coaffinations,
    canonical_clockwork_coaffination,
    clockwork_coaffine_pair,
    find_pair_morphism,
    r_clock,
)
from pycliques.coaffinations import CoaffinePair, is_coaffine_map


def test_r_clock_matches_construction_from_the_paper():
    """R_{2m}^n has 4m crown vertices and 2m(n + 1) core vertices."""
    g = r_clock(2, 1)

    assert g.number_of_nodes() == 4 * 2 + 2 * 2 * (1 + 1)


@pytest.mark.parametrize(
    ("m", "n"),
    [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)],
)
def test_clockwork_coaffine_pair_is_involutive_and_realizes_radius(m, n):
    """The canonical coaffination is an involutive automorphism of radius m+1."""
    pair = clockwork_coaffine_pair(m, n)
    sigma = pair.coaffination

    assert all(sigma[sigma[x]] == x for x in pair.graph)
    distance = dict(nx.all_pairs_shortest_path_length(pair.graph))
    assert min(distance[x][sigma[x]] for x in pair.graph) == m + 1


def test_clockwork_coaffine_pair_rejects_invalid_parameters():
    """m must be positive and n must be nonnegative."""
    with pytest.raises(ValueError):
        clockwork_coaffine_pair(0, 0)
    with pytest.raises(ValueError):
        clockwork_coaffine_pair(1, -1)


def test_canonical_clockwork_coaffination_shifts_segments_by_m():
    """The formula shifts each segment index by m and flips crown positions."""
    sigma = canonical_clockwork_coaffination(1, 0)

    assert sigma == {0: 3, 1: 2, 2: 1, 3: 0, 4: 5, 5: 4}


# ---------------------------------------------------------------------------
# Test 1 -- identity clockwork pair
# ---------------------------------------------------------------------------


def test_find_pair_morphism_finds_identity_clockwork_pair():
    """A coaffine pair maps admissibly onto itself."""
    pair = clockwork_coaffine_pair(1, 0)

    f = find_pair_morphism(pair, pair)

    assert f is not None
    assert is_coaffine_map(pair, pair, f)
    assert all(pair.graph.has_edge(f[u], f[v]) for u, v in pair.graph.edges())


# ---------------------------------------------------------------------------
# Test 3 -- wrong-radius coaffination rejected
# ---------------------------------------------------------------------------


def test_candidate_target_coaffinations_rejects_short_displacement():
    """An involution with displacement < radius is not a candidate."""
    cycle = nx.cycle_graph(4)

    # Swaps adjacent vertices: an automorphism and an involution, but every
    # vertex moves distance 1, which is short of radius 2.
    candidates = list(candidate_target_coaffinations(cycle, 2))

    short_displacement = {0: 1, 1: 0, 2: 3, 3: 2}
    assert short_displacement not in candidates


def test_candidate_target_coaffinations_accepts_correct_displacement():
    """The antipodal map of C4 is the unique radius-2 candidate."""
    cycle = nx.cycle_graph(4)

    candidates = list(candidate_target_coaffinations(cycle, 2))

    assert candidates == [{0: 2, 1: 3, 2: 0, 3: 1}]


# ---------------------------------------------------------------------------
# Test 4 -- equivariance is essential
# ---------------------------------------------------------------------------


def test_find_pair_morphism_requires_equivariance():
    """An ordinary graph morphism exists, but no admissible one does."""
    source = nx.Graph([("a", "b")])
    sigma = {"a": "b", "b": "a"}
    target = nx.Graph([("x", "y")])
    tau = {"x": "x", "y": "y"}

    # An ordinary homomorphism a -> x, b -> y exists (the edge is preserved).
    assert target.has_edge("x", "y")

    f = find_pair_morphism(CoaffinePair(source, sigma), CoaffinePair(target, tau))

    assert f is None


# ---------------------------------------------------------------------------
# Test 5 -- non-injective morphism
# ---------------------------------------------------------------------------


def test_find_pair_morphism_allows_non_injective_maps():
    """A valid admissible morphism may identify nonadjacent source vertices."""
    source = nx.cycle_graph(4)
    sigma = {0: 2, 1: 3, 2: 0, 3: 1}
    target = nx.complete_graph(["A", "B", "C"])
    tau = {"A": "A", "B": "C", "C": "B"}
    source_pair = CoaffinePair(source, sigma)
    target_pair = CoaffinePair(target, tau)

    f = find_pair_morphism(source_pair, target_pair)

    assert f is not None
    assert len(set(f.values())) < len(f)
    assert is_coaffine_map(source_pair, target_pair, f)
    assert all(target.has_edge(f[u], f[v]) for u, v in source.edges())


# ---------------------------------------------------------------------------
# Test 6 -- arbitrary hashable target labels
# ---------------------------------------------------------------------------


def test_arbitrary_hashable_labels_are_supported():
    """Coaffination enumeration and pair-map search work with mixed labels."""
    labels = ["alice", 7, ("carol", 1), "dave"]
    target = nx.relabel_nodes(nx.cycle_graph(4), dict(enumerate(labels)))

    candidates = list(candidate_target_coaffinations(target, 2))
    assert len(candidates) == 1

    source = nx.cycle_graph(4)
    sigma = {0: 2, 1: 3, 2: 0, 3: 1}
    source_pair = CoaffinePair(source, sigma)
    target_pair = CoaffinePair(target, candidates[0])

    f = find_pair_morphism(source_pair, target_pair)

    assert f is not None
    assert set(f.values()) <= set(labels)
    assert is_coaffine_map(source_pair, target_pair, f)
