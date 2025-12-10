import networkx as nx

from pycliques.dominated import (
    closed_neighborhood,
    completely_pared_graph,
    dominates,
    find_dominated_vertex,
    is_dismantlable,
    is_dominated_vertex,
    pared_graph,
    pared_index,
    remove_dominated_vertex,
    twin_classes,
)


def test_closed_neighborhood_returns_neighbors_and_vertex():
    graph = nx.path_graph(4)
    assert closed_neighborhood(graph, 0) == {0, 1}


def test_dominates_detects_subset_relationship():
    graph = nx.path_graph(4)
    assert dominates(graph, 1, 0)
    assert not dominates(graph, 0, 1)


def test_is_dominated_vertex_can_return_dominator():
    graph = nx.path_graph(4)
    assert is_dominated_vertex(graph, 0)
    assert is_dominated_vertex(graph, 0, return_dominator=True) == (True, 1)
    assert not is_dominated_vertex(graph, 2)


def test_find_dominated_vertex_handles_none_case():
    assert find_dominated_vertex(nx.path_graph(4)) == 0
    assert find_dominated_vertex(nx.cycle_graph(4)) is None


def test_remove_dominated_vertex_prunes_one_vertex():
    pruned = remove_dominated_vertex(nx.path_graph(4))
    assert set(pruned.nodes()) == {1, 2, 3}


def test_twin_classes_groups_equivalent_vertices():
    graph = nx.complete_graph(3)
    graph.add_node(3)
    classes = sorted(sorted(cls) for cls in twin_classes(graph))
    assert classes == [[0, 1, 2], [3]]


def test_pared_graph_removes_dominated_representatives():
    pared = pared_graph(nx.path_graph(4))
    assert set(pared.nodes()) == {1, 2}


def test_pared_index_counts_iterations_until_fixed_point():
    assert pared_index(nx.path_graph(4)) == 2


def test_completely_pared_graph_leaves_single_vertex():
    result = completely_pared_graph(nx.path_graph(4))
    assert set(result.nodes()) == {3}


def test_is_dismantlable_distinguishes_graphs():
    assert is_dismantlable(nx.path_graph(4))
    assert not is_dismantlable(nx.cycle_graph(4))
