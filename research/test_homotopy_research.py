"""Reproducible low-order checks for clique-graph homotopy research."""

import networkx as nx
from pycliques import clique_graph
from pycombtop import (
    clique_complex,
    nerve_of_cliques,
    relative_collapse,
    verify_relative_collapse,
)
from pyg6data import graph_generator

from research.homotopy_census import matches_research_filter


def _defect(graph: nx.Graph) -> int:
    kg = clique_graph(graph)
    assert kg is not None
    return len(
        clique_complex(kg).all_simplices()
        - nerve_of_cliques(graph).all_simplices()
    )


def test_defect_one_and_three_absent_in_connected_orders_six_and_seven():
    for order in (6, 7):
        defects = {_defect(graph) for graph in graph_generator(order)}
        assert 1 not in defects
        assert 3 not in defects


def test_defect_two_has_a_single_verified_elementary_pair():
    graph = nx.from_graph6_bytes(b"EElw")
    kg = clique_graph(graph)
    assert kg is not None
    source = clique_complex(kg)
    target = nerve_of_cliques(graph)

    assert len(source.all_simplices() - target.all_simplices()) == 2
    result = relative_collapse(source, target)

    assert result.status.value == "PROVED"
    assert len(result.pairs) == 1
    assert verify_relative_collapse(source, target, result.pairs)


def test_paper_candidate_filter_counts_orders_seven_and_eight():
    expected = {7: 2, 8: 34}
    for order, expected_count in expected.items():
        count = 0
        for graph in graph_generator(order):
            kg = clique_graph(graph)
            assert kg is not None
            count += matches_research_filter(graph, kg)
        assert count == expected_count
