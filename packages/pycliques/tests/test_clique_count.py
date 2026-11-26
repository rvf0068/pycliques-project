"""Exact maximal-clique counting without listing maximal cliques."""

import networkx as nx
import pytest
from pycliques import count_maximal_cliques


def test_count_against_graph_atlas():
    for graph in nx.graph_atlas_g():
        assert count_maximal_cliques(graph) == sum(1 for _ in nx.find_cliques(graph))


def test_exponential_count_and_limits():
    graph = nx.complete_multipartite_graph(*([2] * 24))
    assert count_maximal_cliques(graph) == 2**24
    assert count_maximal_cliques(graph, max_states=1) is None
    assert count_maximal_cliques(nx.complete_graph(4), max_cache=0) == 1
    induced = graph.subgraph(range(20))
    assert count_maximal_cliques(induced) == 2**10


def test_count_validation_and_labels():
    graph = nx.cycle_graph(["a", "b", "c", "d", "e"])
    assert count_maximal_cliques(graph) == 5
    with pytest.raises(ValueError):
        count_maximal_cliques(nx.DiGraph())
    for kwargs in ({"max_states": 0}, {"max_cache": -1}, {"max_states": True}):
        with pytest.raises(ValueError):
            count_maximal_cliques(graph, **kwargs)
