import networkx as nx
import pytest
from pycliques.cutpoints import (
    contract_local_bridge,
    edge_in_triangle,
    edges_in_no_triangle,
    is_local_bridge,
    local_bridge_contractions,
    local_bridges,
    non_triangle_edge_removals,
    remove_edge_not_in_triangle,
)


def test_local_bridge_on_path_and_bridge():
    graph = nx.path_graph(4)

    assert is_local_bridge(graph, 1, 2)
    assert is_local_bridge(graph, 0, 1)


def test_short_alternate_path_is_not_local_bridge():
    assert not is_local_bridge(nx.cycle_graph(4), 0, 1)


def test_triangle_edge_is_not_local_bridge():
    assert not is_local_bridge(nx.complete_graph(3), 0, 1)


def test_non_edge_is_not_local_bridge():
    assert not is_local_bridge(nx.path_graph(3), 0, 2)


def test_local_bridge_labels_and_contraction():
    graph = nx.Graph([("left", "right"), ("right", "tail")])

    contracted = contract_local_bridge(graph, "left", "right")

    assert set(contracted) == {"left", "tail"}
    assert contracted.has_edge("left", "tail")
    assert set(graph) == {"left", "right", "tail"}


def test_local_bridges_and_contractions():
    graph = nx.path_graph(4)

    assert set(local_bridges(graph)) == {(0, 1), (1, 2), (2, 3)}
    assert len(list(local_bridge_contractions(graph))) == 3


def test_contract_local_bridge_rejects_invalid_edges():
    with pytest.raises(ValueError, match="not an edge"):
        contract_local_bridge(nx.path_graph(3), 0, 2)
    with pytest.raises(ValueError, match="not a local bridge"):
        contract_local_bridge(nx.cycle_graph(4), 0, 1)


def test_triangle_edges_are_not_removable():
    graph = nx.complete_graph(3)

    assert edge_in_triangle(graph, 0, 1)
    assert list(edges_in_no_triangle(graph)) == []
    with pytest.raises(ValueError, match="contained in a triangle"):
        remove_edge_not_in_triangle(graph, 0, 1)


def test_path_edges_are_removable_and_labels_are_preserved():
    graph = nx.Graph([(("a", 1), ("b", 2)), (("b", 2), ("c", 3))])

    assert not edge_in_triangle(graph, ("a", 1), ("b", 2))
    assert set(edges_in_no_triangle(graph)) == {
        (("a", 1), ("b", 2)),
        (("b", 2), ("c", 3)),
    }
    removed = remove_edge_not_in_triangle(graph, ("a", 1), ("b", 2))
    assert set(removed) == set(graph)
    assert not removed.has_edge(("a", 1), ("b", 2))
    assert graph.has_edge(("a", 1), ("b", 2))


def test_non_triangle_edge_removals():
    assert len(list(non_triangle_edge_removals(nx.path_graph(3)))) == 2


def test_icosahedron_plus_antipodal_edge():
    graph = nx.icosahedral_graph()
    antipodes = (0, 3)
    graph.add_edge(*antipodes)

    assert not edge_in_triangle(graph, *antipodes)
    removed = remove_edge_not_in_triangle(graph, *antipodes)
    assert nx.is_isomorphic(removed, nx.icosahedral_graph())


def test_icosahedron_plus_degree_two_vertex():
    graph = nx.icosahedral_graph()
    graph.add_edges_from([(12, 0), (12, 3)])

    assert not edge_in_triangle(graph, 12, 0)
    removed = remove_edge_not_in_triangle(graph, 12, 0)
    assert removed.has_edge(12, 3)
    assert removed.degree(12) == 1
