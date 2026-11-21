"""Tests for reusable octahedral search and amplification, without snub seeds."""

import networkx as nx
import pycliques
import pytest
from pycliques.octahedral_amplification import (
    amplify_induced_octahedron,
    augment_induced_octahedron,
    find_induced_octahedron,
    is_dominating_octahedron,
)


def _pairs(dimension: int) -> list[tuple[int, int]]:
    return [(2 * i, 2 * i + 1) for i in range(dimension)]


def test_independent_search_and_dominating_certificates() -> None:
    graph = pycliques.octahedron(4)
    pairs = find_induced_octahedron(graph, 4)
    assert pairs is not None
    assert is_dominating_octahedron(graph, pairs)
    assert find_induced_octahedron(graph, 5, max_states=100) is None
    smaller = find_induced_octahedron(graph, 3)
    assert smaller is not None
    assert not is_dominating_octahedron(graph, smaller)


def test_exact_amplification_from_octahedron() -> None:
    host = pycliques.octahedron(3)
    pairs = _pairs(3)
    target_pairs = amplify_induced_octahedron(host, pairs)
    assert len(target_pairs) == 4
    target_vertices = {v for pair in target_pairs for v in pair}
    clique_graph = pycliques.clique_graph(host)
    assert clique_graph is not None
    assert set(clique_graph) == target_vertices
    assert nx.is_isomorphic(
        clique_graph.subgraph(target_vertices), pycliques.octahedron(4)
    )
    assert is_dominating_octahedron(host, pairs)


def test_amplification_rejects_universal_outside_neighbor() -> None:
    host = pycliques.octahedron(3)
    host.add_edges_from((6, v) for v in range(6))
    assert not is_dominating_octahedron(host, _pairs(3))
    with pytest.raises(ValueError, match="adjacent to all"):
        amplify_induced_octahedron(host, _pairs(3))


def test_one_for_two_augments_smaller_octahedron() -> None:
    graph = pycliques.octahedron(4)
    grown = augment_induced_octahedron(
        graph, _pairs(3), max_walk_steps=0
    )
    assert grown is not None
    assert len(grown) == 4
    assert is_dominating_octahedron(graph, grown)
    assert augment_induced_octahedron(
        pycliques.octahedron(3), _pairs(3), max_walk_steps=0
    ) is None


def test_augmentation_can_be_non_dominating() -> None:
    graph = pycliques.octahedron(5)
    grown = augment_induced_octahedron(
        graph, _pairs(3), max_walk_steps=0
    )
    assert grown is not None
    assert len(grown) == 4
    assert not is_dominating_octahedron(graph, grown)
    assert augment_induced_octahedron(
        graph, _pairs(3), max_walk_steps=0, require_dominating=True
    ) is None


def test_augmented_octahedron_handles_arbitrary_labels() -> None:
    graph = nx.relabel_nodes(
        pycliques.octahedron(4),
        {i: f"v{i}" for i in range(8)},
    )
    start = [(f"v{i}", f"v{i + 1}") for i in (0, 2, 4)]
    result = augment_induced_octahedron(
        graph, start, max_walk_steps=0
    )
    assert result is not None
    assert is_dominating_octahedron(graph, result)


@pytest.mark.parametrize(
    "pairs",
    [
        [(0, 1), (0, 3), (4, 5)],
        [(0, 2), (2, 3), (4, 5)],
        [(0, 1), (2, 3)],
    ],
)
def test_bad_octahedral_pairs(pairs: list[tuple[int, int]]) -> None:
    graph = pycliques.octahedron(3)
    if pairs == [(0, 1), (2, 3)]:
        assert is_dominating_octahedron(graph, pairs) is False
        return
    with pytest.raises(ValueError):
        amplify_induced_octahedron(graph, pairs)


@pytest.mark.parametrize(
    "operation",
    ["amplify", "augment", "search"],
)
def test_directed_graph_rejected(operation: str) -> None:
    graph = nx.DiGraph(pycliques.octahedron(3))
    with pytest.raises(ValueError, match="undirected simple loopless"):
        if operation == "search":
            find_induced_octahedron(graph, 3)
        elif operation == "augment":
            augment_induced_octahedron(graph, _pairs(3))
        else:
            amplify_induced_octahedron(graph, _pairs(3))


def test_explicit_resource_limits() -> None:
    with pytest.raises(ValueError, match="max_facets"):
        amplify_induced_octahedron(
            pycliques.octahedron(3), _pairs(3), max_facets=7
        )
    with pytest.raises(ValueError, match="max_checks"):
        augment_induced_octahedron(
            pycliques.octahedron(3), _pairs(3), max_checks=0
        )
    with pytest.raises(ValueError, match="max_states"):
        find_induced_octahedron(pycliques.octahedron(3), 3, max_states=0)
