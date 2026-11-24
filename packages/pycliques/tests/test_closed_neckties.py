"""Cross-check bitset closure and centers against direct enumeration."""

from itertools import combinations

import networkx as nx
import pytest
from pycliques.closed_neckties import ClosedNecktieIndex, check_closed_centers
from pycliques.homotopy_invariance import is_center
from pycliques.implicit_clique_graph import ImplicitCliqueGraph
from pycliques.poset_homotopy import h_closure_from_neighbors


@pytest.mark.parametrize(
    "family",
    [
        [{0}],
        [{0, 1}, {1, 2}, {2, 0}],
        [{0, 1}, {1, 2}, {2, 3}, {3, 0}],
        [{0, 1, 2}, {0, 1, 3}, {0, 2, 4}, {1, 3, 4}, {2, 3, 4}],
        list(map(frozenset, nx.find_cliques(nx.octahedral_graph()))),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_against_explicit_completes(family, lazy):
    index = ClosedNecktieIndex(family, lazy=lazy, cache_size=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(family)))
    graph.add_edges_from(
        (i, j) for i, j in combinations(graph, 2) if set(family[i]) & set(family[j])
    )
    closed_sets = set()
    for complete in nx.enumerate_all_cliques(graph):
        mask = sum(1 << i for i in complete)
        h = h_closure_from_neighbors(graph, complete)
        expected = sum(1 << i for i in h)
        assert index.close(mask) == expected
        closed_sets.add(expected)
        assert index.centers(mask) == sum(
            1 << i for i in h if is_center(family[i], [family[j] for j in complete])
        )
        intersection = set.intersection(*(set(family[j]) for j in complete))
        assert index.is_bad(mask) == (not intersection)
    for anchor in graph:
        got = list(index.closed_containing(anchor))
        assert len(got) == len(set(got))
        assert set(got) == {m for m in closed_sets if m & (1 << anchor)}
    assert set(index.closed_representatives()) == closed_sets
    assert check_closed_centers(index).holds == all(
        not index.is_bad(m) or bool(index.centers(m)) for m in closed_sets
    )


def test_symmetry_and_limits():
    index = ClosedNecktieIndex([{0, 1}, {1, 2}, {2, 0}])
    result = check_closed_centers(index, generators=[(1, 2, 0)])
    assert result.holds is False
    assert result.witness == 7
    assert index.members(7) == index.family
    assert set(index.nerve_facets(7)) == {3, 5, 6}
    assert check_closed_centers(index, max_checks=0).holds is None
    assert check_closed_centers(ClosedNecktieIndex([])).holds is True
    with pytest.raises(ValueError):
        list(index.closed_representatives([(0, 0, 1)]))
    asymmetric = ClosedNecktieIndex([{0}, {0, 1}, {1, 2}])
    with pytest.raises(ValueError):
        list(asymmetric.closed_representatives([(1, 0, 2)]))
    with pytest.raises(ValueError):
        check_closed_centers(index, max_checks=-1)


def test_bad_input():
    with pytest.raises(ValueError):
        ClosedNecktieIndex([[]])
    with pytest.raises(ValueError):
        ClosedNecktieIndex([{0}, {0}])
    index = ClosedNecktieIndex([{0}, {1}])
    for mask in (0, -1, 4, 3):
        with pytest.raises(ValueError):
            index.close(mask)
    with pytest.raises(ValueError):
        list(index.closed_containing(2))


@pytest.mark.parametrize(
    "family",
    [
        [],
        [{0}],
        [{0, 1}, {1, 2}, {2, 3}],
        [{0, 1}, {1, 2}, {2, 0}],
        list(map(frozenset, nx.find_cliques(nx.octahedral_graph()))),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_restricted_closure_and_maximal_completes(family, lazy):
    index = ClosedNecktieIndex(family, lazy=lazy, cache_size=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(family)))
    graph.add_edges_from(
        (i, j) for i, j in combinations(graph, 2) if family[i] & family[j]
    )
    maximal = {sum(1 << i for i in q) for q in nx.find_cliques(graph)}
    assert set(index.maximal_complete_orbits()) == {(q, 1) for q in maximal}
    masks = [sum(1 << i for i in q) for q in nx.enumerate_all_cliques(graph)]
    all_closed = {index.close(mask) for mask in masks}
    for q in maximal:
        expected = {m for m in all_closed if m & q == m}
        got = list(index.closed_within(q))
        assert set(got) == expected
        assert len(got) == len(expected)

        intervals = []

        def prune(m, e):
            assert m & e == m and e & q == e
            if m.bit_count() >= 2:
                intervals.append((m, e))
                return True
            return False

        remaining = set(index.closed_within(q, prune=prune))
        covered = set()
        for a, b in intervals:
            covered.update(m for m in expected if a & m == a and m & b == m)
        assert remaining | covered == expected


def test_maximal_complete_orbits_with_symmetry():
    index = ClosedNecktieIndex([{0, 1}, {1, 2}, {2, 3}, {3, 0}])
    assert list(index.maximal_complete_orbits([(1, 2, 3, 0)])) == [(3, 4)]
    with pytest.raises(ValueError):
        list(index.maximal_complete_orbits([(1, 0, 2, 3)]))
    with pytest.raises(ValueError):
        list(index.closed_within(1))
    with pytest.raises(ValueError):
        list(index.closed_within(5))


def test_shared_implicit_index_and_anchored_pruning():
    family = [{0, 1}, {1, 2}, {2, 3}, {3, 0}]
    implicit = ImplicitCliqueGraph.from_family(family)
    index = ClosedNecktieIndex.from_implicit(implicit, cache_size=0)
    assert tuple(index.family) == tuple(map(frozenset, family))
    assert index.family[-1] == family[-1]
    assert index.family[:2] == tuple(map(frozenset, family[:2]))
    assert index.adjacency[0] == 11
    with pytest.raises(IndexError):
        _ = index.family[4]
    with pytest.raises(IndexError):
        _ = index.family[-5]
    stars = zip(implicit.base_vertices, implicit.incidence_masks, strict=True)
    assert index.stars == dict(stars)
    assert list(index.vertex_orbit_representatives([(1, 2, 3, 0)])) == [0]
    expected = set(index.closed_containing(0))
    intervals = []

    def prune(m, e):
        intervals.append((m, e))
        return True

    assert not list(index.closed_containing(0, prune=prune))
    for m in expected:
        assert any(a & m == a and m & b == m for a, b in intervals)
    pared = ImplicitCliqueGraph.from_family([{0}, {0, 1}]).pare().core
    with pytest.raises(ValueError):
        ClosedNecktieIndex.from_implicit(pared)
    with pytest.raises(ValueError):
        ClosedNecktieIndex(family, lazy=True, cache_size=-1)


def test_anchored_pruning_covers_all_small_graph_traces():
    # Use literal intersections of maximal completes as the independent
    # oracle; do not call the implementation's closure to build expectations.
    for base in nx.graph_atlas_g():
        if len(base) > 5:
            break
        family = list(map(frozenset, nx.find_cliques(base)))
        graph = nx.Graph()
        graph.add_nodes_from(range(len(family)))
        graph.add_edges_from(
            (i, j) for i, j in combinations(graph, 2) if family[i] & family[j]
        )
        maximal = [set(q) for q in nx.find_cliques(graph)]
        expected = set()
        for complete in nx.enumerate_all_cliques(graph):
            containing = [q for q in maximal if set(complete) <= q]
            closed = set.intersection(*containing)
            expected.add(sum(1 << i for i in closed))
        for cache_size in (0, 2):
            index = ClosedNecktieIndex(family, lazy=True, cache_size=cache_size)
            for anchor in graph:
                want = {m for m in expected if m & (1 << anchor)}
                got = list(index.closed_containing(anchor))
                assert set(got) == want
                assert len(got) == len(want)
                intervals = []

                def prune(m, e):
                    assert m & e == m
                    assert m in want
                    if m.bit_count() >= 2 and m.bit_count() % 2 == 0:
                        intervals.append((m, e))
                        return True
                    return False

                remaining = set(index.closed_containing(anchor, prune=prune))
                assert remaining <= want
                for m in want - remaining:
                    assert any(a & m == a and m & b == m for a, b in intervals)


def test_implicit_index_rejects_negative_cache():
    graph = ImplicitCliqueGraph.from_family([{0}])
    with pytest.raises(ValueError):
        ClosedNecktieIndex.from_implicit(graph, cache_size=-1)
