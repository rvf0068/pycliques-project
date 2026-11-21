"""Focused checks for octahedral search guards and bounded branches."""

import networkx as nx
import pycliques
import pytest
from pycliques.octahedral_amplification import (
    _check_pairs,
    amplify_induced_octahedron,
    augment_induced_octahedron,
    find_induced_octahedron,
    is_dominating_octahedron,
)


def _pairs(dimension: int) -> list[tuple[int, int]]:
    return [(2 * i, 2 * i + 1) for i in range(dimension)]


@pytest.mark.parametrize("kind", ["multigraph", "self_loop"])
def test_rejects_other_invalid_graphs(kind: str) -> None:
    graph: nx.Graph = nx.MultiGraph() if kind == "multigraph" else nx.Graph()
    graph.add_edges_from(((0, 1), (1, 2)))
    if kind == "self_loop":
        graph.add_edge(0, 0)
    with pytest.raises(ValueError, match="undirected simple loopless"):
        find_induced_octahedron(graph, 2)


@pytest.mark.parametrize(
    "pairs,match",
    [
        ([], "at least two"),
        ([(0, 1)], "at least two"),
        ([(0,), (2, 3)], "at least two"),
        ([(0, 1), (2, 3), (4, 8)], "distinct host vertices"),
        ([(0, 1), (0, 3), (4, 5)], "distinct host vertices"),
        ([(0, 2), (1, 3), (4, 5)], "nonadjacent"),
    ],
)
def test_invalid_pair_specs(pairs: list[tuple[int, ...]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _check_pairs(pycliques.octahedron(3), pairs)


def test_missing_cross_edge_is_rejected() -> None:
    graph = pycliques.octahedron(3)
    graph.remove_edge(0, 2)
    with pytest.raises(ValueError, match="completely joined"):
        is_dominating_octahedron(graph, _pairs(3))


@pytest.mark.parametrize("dimension", [0, 1, -1, True])
def test_search_rejects_bad_dimension(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension"):
        find_induced_octahedron(pycliques.octahedron(3), dimension)


def test_search_refuses_boolean_budget() -> None:
    with pytest.raises(ValueError, match="max_states"):
        find_induced_octahedron(pycliques.octahedron(3), 3, max_states=True)


def test_search_dominating_requirement_and_bounded_backtrack() -> None:
    graph = pycliques.octahedron(4)
    assert find_induced_octahedron(graph, 3, require_dominating=True) is None
    assert find_induced_octahedron(graph, 4, max_states=1) is None
    assert find_induced_octahedron(nx.complete_graph(5), 2) is None


def test_amplification_greedily_extends_facets() -> None:
    graph = pycliques.octahedron(3)
    graph.add_edges_from((6, v) for v in (0, 2, 4))
    assert is_dominating_octahedron(graph, _pairs(3))
    amplified = amplify_induced_octahedron(graph, _pairs(3))
    assert len(amplified) == 4
    assert any(6 in facet for opposite_pair in amplified for facet in opposite_pair)
    all_facets = [facet for opposite_pair in amplified for facet in opposite_pair]
    assert all(
        not any(
            u not in facet and all(graph.has_edge(u, v) for v in facet)
            for u in graph
        )
        for facet in all_facets
    )


@pytest.mark.parametrize("max_facets", [0, -1, True])
def test_amplification_rejects_invalid_facets_budget(max_facets: int) -> None:
    with pytest.raises(ValueError, match="max_facets"):
        amplify_induced_octahedron(
            pycliques.octahedron(3), _pairs(3), max_facets=max_facets
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_walk_steps": -1}, "max_walk_steps"),
        ({"max_walk_steps": True}, "max_walk_steps"),
        ({"max_checks": True}, "max_checks"),
        ({"seed": "not-an-integer"}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_augmentation_rejects_invalid_parameters(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        augment_induced_octahedron(pycliques.octahedron(3), _pairs(3), **kwargs)


def test_augmentation_random_walk_without_possible_growth() -> None:
    graph = pycliques.octahedron(4)
    assert (
        augment_induced_octahedron(
            graph, _pairs(4), max_walk_steps=2, max_checks=200, seed=33
        )
        is None
    )


def test_augmentation_checks_expire_during_two_pair_search() -> None:
    graph = pycliques.octahedron(4)
    assert (
        augment_induced_octahedron(graph, _pairs(3), max_walk_steps=0, max_checks=1)
        is None
    )


def test_augmentation_checks_expire_during_random_walk() -> None:
    graph = pycliques.octahedron(4)
    assert (
        augment_induced_octahedron(graph, _pairs(4), max_walk_steps=1, max_checks=4)
        is None
    )
