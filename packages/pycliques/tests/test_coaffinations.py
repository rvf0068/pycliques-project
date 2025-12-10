import networkx as nx

from pycliques.cliques import clique_graph
from pycliques.coaffinations import CoaffinePair, automorphisms, coaffinations


def test_automorphisms_cycle_graph_has_six():
    graph = nx.cycle_graph(3)
    autos = list(automorphisms(graph))
    assert len(autos) == 6
    assert all(set(auto.keys()) == set(graph.nodes) for auto in autos)


def test_coaffinations_enforce_minimum_distance():
    graph = nx.octahedral_graph()
    coafs = list(coaffinations(graph, 2))
    assert len(coafs) == 1
    mapping = coafs[0]
    assert all(graph.degree[v] == graph.degree[mapping[v]] for v in graph)


def test_clique_graph_supports_coaffine_pair():
    graph = nx.cycle_graph(4)
    pair = CoaffinePair(graph, {0: 2, 1: 3, 2: 0, 3: 1})
    result = clique_graph(pair)
    assert isinstance(result, CoaffinePair)
    assert result.graph.number_of_nodes() == 4
    assert set(result.coaffination.keys()) == set(result.graph.nodes)
