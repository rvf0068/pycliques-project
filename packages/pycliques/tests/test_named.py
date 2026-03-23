import networkx as nx
from pycliques.named import (
    complement_of_cycle,
    graph_suspension,
    octahedron,
    snub_dysphenoid,
    suspension_of_cycle,
)


def test_graph_suspension_adds_two_universal_vertices():
    g = graph_suspension(nx.empty_graph(3))
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 6


def test_graph_suspension_of_single_vertex():
    g = graph_suspension(nx.empty_graph(1))
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


def test_suspension_of_cycle_4_is_octahedral():
    assert nx.is_isomorphic(nx.octahedral_graph(), suspension_of_cycle(4))


def test_suspension_of_cycle_preserves_order():
    g = suspension_of_cycle(5)
    assert g.number_of_nodes() == 7


def test_complement_of_cycle_node_count():
    g = complement_of_cycle(5)
    assert g.number_of_nodes() == 5


def test_complement_of_cycle_edge_count():
    # C_5 has 5 edges; complement has C(5,2) - 5 = 5 edges
    assert complement_of_cycle(5).number_of_edges() == 5


def test_octahedron_3_is_octahedral():
    assert nx.is_isomorphic(nx.octahedral_graph(), octahedron(3))


def test_octahedron_complement_is_disjoint_edges():
    comp = nx.complement(octahedron(4))
    assert comp.number_of_edges() == 4
    assert all(d == 1 for _, d in comp.degree())


def test_snub_dysphenoid_has_8_vertices():
    g = snub_dysphenoid()
    assert g.number_of_nodes() == 8


def test_snub_dysphenoid_is_connected():
    assert nx.is_connected(snub_dysphenoid())
