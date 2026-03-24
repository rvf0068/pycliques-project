import networkx as nx
import pytest
from pycliques.retractions import (
    _is_maximal_clique,
    _string_to_graph,
    dict_to_tuple,
    graph_from_gap_adjacency_list,
    has_induced,
    invert_dict,
    is_map,
    retraction,
    retracts,
    retracts_to,
    special_octahedra,
)


def test_dict_to_tuple_preserves_content():
    result = dict_to_tuple({1: "a", 2: "b"})
    assert result == ((1, "a"), (2, "b"))


def test_invert_dict_swaps_keys_and_values():
    assert invert_dict({0: "a", 1: "b"}) == {"a": 0, "b": 1}


def test_graph_from_gap_adjacency_list_shifts_indices():
    g = graph_from_gap_adjacency_list([[2], [1]])
    assert sorted(g.edges()) == [(0, 1)]


def test_graph_from_gap_adjacency_list_triangle():
    g = graph_from_gap_adjacency_list([[2, 3], [1, 3], [1, 2]])
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3


def test_is_map_valid_homomorphism():
    mapping = {0: 0, 1: 1, 2: 0, 3: 1}
    assert is_map(nx.cycle_graph(4), nx.complete_graph(2), mapping)


def test_is_map_partial_mapping():
    mapping = {0: 0, 1: 1}
    assert is_map(nx.cycle_graph(4), nx.complete_graph(2), mapping)


def test_is_map_invalid_homomorphism():
    # Map adjacent vertices to non-adjacent vertices in a path
    mapping = {0: 0, 1: 2}
    assert not is_map(nx.complete_graph(2), nx.path_graph(3), mapping)


def test_retraction_path_to_edge():
    rets = list(retraction(nx.path_graph(3), nx.path_graph(2)))
    assert len(rets) == 2
    for ret_map, incl_map in rets:
        assert is_map(nx.path_graph(3), nx.path_graph(2), ret_map)


def test_retraction_wheel_to_cycle_is_empty():
    assert list(retraction(nx.wheel_graph(4), nx.cycle_graph(4))) == []


def test_retracts_finds_retraction():
    result = retracts(nx.path_graph(3), nx.path_graph(2))
    assert result is not None


def test_retracts_returns_none_when_impossible():
    result = retracts(nx.wheel_graph(4), nx.cycle_graph(4))
    assert result is None


def test_retracts_to_returns_callable():
    checker = retracts_to(nx.path_graph(2))
    assert checker(nx.path_graph(3)) is not None
    # An empty graph with one vertex cannot retract to an edge
    assert checker(nx.empty_graph(1)) is None


def test_has_induced_finds_subgraph():
    # C5 contains P3 as an induced subgraph
    result = has_induced(nx.cycle_graph(5), nx.path_graph(3))
    assert result is not None


def test_has_induced_returns_none_when_absent():
    result = has_induced(nx.cycle_graph(4), nx.complete_graph(3))
    assert result is None


def test_special_octahedra_on_octahedral_graph():
    assert special_octahedra(nx.octahedral_graph()) is True


def test_special_octahedra_on_cycle():
    assert special_octahedra(nx.cycle_graph(5)) is False


def test_special_octahedra_on_complete_graph():
    # K4 has no induced octahedron
    assert special_octahedra(nx.complete_graph(4)) is False


def test_is_maximal_clique_true():
    # In a triangle, the full triangle is maximal
    g = nx.complete_graph(3)
    assert _is_maximal_clique(g, [0, 1, 2]) is True


def test_is_maximal_clique_false():
    # In K4, a triangle is not maximal
    g = nx.complete_graph(4)
    assert _is_maximal_clique(g, [0, 1, 2]) is False


def test_is_maximal_clique_edge_in_path():
    g = nx.path_graph(4)  # 0-1-2-3
    # {0,1} is a maximal clique in a path
    assert _is_maximal_clique(g, [0, 1]) is True


def test_string_to_graph_suspension_of_cycle():
    g = _string_to_graph("sc5")
    # Suspension of C5 has 5+2=7 vertices
    assert g.number_of_nodes() == 7


def test_string_to_graph_complement_of_cycle():
    g = _string_to_graph("cc5")
    assert g.number_of_nodes() == 5


def test_string_to_graph_octahedron():
    g = _string_to_graph("o3")
    # Octahedron(3) = complement of 3 disjoint edges = 6 vertices
    assert g.number_of_nodes() == 6


def test_string_to_graph_invalid_raises():
    with pytest.raises(ValueError, match="Unknown graph string format"):
        _string_to_graph("xyz")
