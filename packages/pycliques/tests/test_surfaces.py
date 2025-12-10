import networkx as nx

from pycliques.surfaces import (
    is_closed_surface,
    is_cycle,
    is_path,
    is_regular,
    is_surface,
    open_neighborhood,
)


def test_is_regular_on_cycle_and_path():
    assert is_regular(nx.cycle_graph(4), 2)
    assert not is_regular(nx.path_graph(4), 2)


def test_is_cycle_requires_connected_degree_two():
    assert is_cycle(nx.cycle_graph(5))
    assert not is_cycle(nx.path_graph(5))


def test_is_path_recognizes_simple_path():
    assert is_path(nx.path_graph(4))
    assert not is_path(nx.complete_graph(3))


def test_open_neighborhood_induced_subgraph():
    graph = nx.cycle_graph(4)
    neighborhood = open_neighborhood(graph, 0)
    assert set(neighborhood.nodes()) == {1, 3}
    assert neighborhood.number_of_edges() == 0


def test_is_closed_surface_on_tetrahedral_graph():
    sphere = nx.octahedral_graph()
    assert is_closed_surface(sphere)
    assert not is_closed_surface(nx.cycle_graph(4))


def test_is_surface_accepts_path_neighborhoods():
    graph = nx.diamond_graph()
    assert is_surface(graph)
    assert not is_surface(nx.cycle_graph(4))
