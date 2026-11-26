"""Exact strong-product coordinates, including ambiguous Cartesian classes."""

from dataclasses import replace

import networkx as nx
import pytest
from pycliques import (
    StrongProductSplit,
    find_strong_product_split,
    strong_product_factors,
    verify_strong_product_split,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (nx.cycle_graph(5), nx.cycle_graph(7)),
        (nx.cycle_graph(4), nx.cycle_graph(4)),
        (nx.path_graph(3), nx.cycle_graph(5)),
        (nx.path_graph(4), nx.path_graph(4)),
        (nx.complete_bipartite_graph(2, 3), nx.cycle_graph(5)),
    ],
)
def test_split_reconstructs_strong_product(left, right):
    graph = nx.strong_product(left, right)
    graph = nx.relabel_nodes(graph, {v: str(v) for v in graph})
    split = find_strong_product_split(graph)
    assert split is not None
    assert verify_strong_product_split(graph, split)
    a, b = strong_product_factors(graph, split)
    assert nx.is_isomorphic(nx.strong_product(a, b), graph)


def test_cartesian_classes_must_be_grouped():
    graph = nx.strong_product(nx.cycle_graph(4), nx.cycle_graph(4))
    assert find_strong_product_split(graph, max_partitions=1) is None
    assert find_strong_product_split(graph) is not None
    assert find_strong_product_split(graph, max_vertices=15) is None


@pytest.mark.parametrize(
    "graph", [nx.cycle_graph(5), nx.cycle_graph(6), nx.complete_graph(4)]
)
def test_inconclusive_search(graph):
    assert find_strong_product_split(graph) is None


def test_invalid_search_inputs():
    for graph in (
        nx.Graph(),
        nx.empty_graph(2),
        nx.DiGraph([(0, 1)]),
        nx.MultiGraph([(0, 1)]),
        nx.Graph([(0, 0)]),
    ):
        with pytest.raises(ValueError):
            find_strong_product_split(graph)
    with pytest.raises(ValueError):
        find_strong_product_split(nx.path_graph(4), max_partitions=0)
    with pytest.raises(ValueError):
        find_strong_product_split(nx.path_graph(4), max_vertices=0)


def test_verify_coordinates_not_merely_a_cartesian_product():
    a, b = nx.cycle_graph(4), nx.path_graph(3)
    graph = nx.strong_product(a, b)
    split = StrongProductSplit({v: v for v in graph})
    assert verify_strong_product_split(graph, split)
    cartesian = nx.cartesian_product(a, b)
    assert not verify_strong_product_split(cartesian, split)
    graph.remove_edge((0, 0), (1, 1))
    assert not verify_strong_product_split(graph, split)
    assert not verify_strong_product_split(nx.Graph(), split)
    assert not verify_strong_product_split(graph, None)


def test_malformed_coordinates():
    graph = nx.strong_product(nx.path_graph(3), nx.path_graph(3))
    split = StrongProductSplit({v: v for v in graph})
    assert verify_strong_product_split(graph, split)
    for value in ((-1, 0), (True, 0), (1,), [0, 0], (1, 0)):
        coords = dict(split.coordinates)
        coords[(0, 0)] = value
        assert not verify_strong_product_split(
            graph, replace(split, coordinates=coords)
        )
    coords = dict(split.coordinates)
    del coords[(0, 0)]
    assert not verify_strong_product_split(graph, replace(split, coordinates=coords))
    assert not verify_strong_product_split(
        graph, StrongProductSplit({v: (i, 0) for i, v in enumerate(graph)})
    )


def test_thick_product_explicit_coordinates():
    graph = nx.strong_product(nx.complete_graph(2), nx.cycle_graph(5))
    assert find_strong_product_split(graph) is None
    assert verify_strong_product_split(graph, StrongProductSplit({v: v for v in graph}))
