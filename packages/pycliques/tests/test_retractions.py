import networkx as nx
from pycliques.retractions import (
    dict_to_tuple,
    graph_from_gap_adjacency_list,
    has_induced,
    invert_dict,
    is_map,
    retraction,
    retracts,
    retracts_to,
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
