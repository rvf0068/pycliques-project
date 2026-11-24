"""Cross-check bitset closure and centers against direct enumeration."""

from itertools import combinations

import networkx as nx
import pytest
from pycliques.closed_neckties import ClosedNecktieIndex, check_closed_centers
from pycliques.homotopy_invariance import is_center
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
def test_against_explicit_completes(family):
    index = ClosedNecktieIndex(family)
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
