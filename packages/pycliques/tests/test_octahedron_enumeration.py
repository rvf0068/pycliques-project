"""Exact enumeration of induced octahedra, including bounded exits."""

from itertools import combinations

import networkx as nx
import pytest
from pycliques import enumerate_induced_octahedra, verify_induced_octahedron


def test_enumeration_matches_exhaustive_subsets():
    for graph in nx.graph_atlas_g():
        if len(graph) > 6:
            break
        for d in (1, 2, 3):
            search = enumerate_induced_octahedra(graph, target_dimension=d)
            actual = []
            for pairs in search:
                assert verify_induced_octahedron(graph, pairs)
                actual.append(frozenset(v for pair in pairs for v in pair))
            expected = {
                frozenset(s)
                for s in combinations(graph, 2 * d)
                if all(k == 2 * d - 2 for _, k in graph.subgraph(s).degree())
            }
            assert len(actual) == len(set(actual))
            assert set(actual) == expected
            assert search.exhausted and not search.limited


def test_early_stop_and_limits():
    graph = nx.complete_multipartite_graph(*([2] * 8))
    search = enumerate_induced_octahedra(graph, target_dimension=4)
    assert len(next(search)) == 4
    assert not search.exhausted
    assert len(list(search)) == 69
    assert search.exhausted
    small = enumerate_induced_octahedra(graph, target_dimension=4, max_states=1)
    assert list(small) == []
    assert small.limited and not small.exhausted
    with pytest.raises(ValueError):
        enumerate_induced_octahedra(nx.DiGraph(), target_dimension=4)
    with pytest.raises(ValueError):
        enumerate_induced_octahedra(graph, target_dimension=0)
    with pytest.raises(ValueError):
        enumerate_induced_octahedra(graph, target_dimension=4, max_states=0)


def test_enumeration_resource_error_is_not_exhaustion(monkeypatch):
    def unavailable(_):
        raise RecursionError("simulated recursion limit")

    graph = nx.complete_multipartite_graph(2, 2, 2)
    search = enumerate_induced_octahedra(graph, target_dimension=3)
    monkeypatch.setattr("pycliques.octahedral_nonlocal_search._bits", unavailable)
    assert list(search) == []
    assert search.limited and not search.exhausted
