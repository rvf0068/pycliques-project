"""Blind compatibility-search tests, including the snub with no saved pairs."""

from __future__ import annotations

import networkx as nx
import pycliques
import pytest


def test_compatibility_search_discovers_o4_with_arbitrary_labels() -> None:
    original = nx.complete_multipartite_graph(2, 2, 2, 2)
    graph = nx.relabel_nodes(original, {v: ("vertex", v) for v in original})
    result = pycliques.find_induced_octahedron_compatibility(
        graph, target_dimension=4, max_restarts=10, seed=11
    )
    assert result.status is pycliques.OctahedronSearchStatus.FOUND
    assert result.restarts >= 1
    assert result.antipodal_pairs is not None
    assert len(result.antipodal_pairs) >= 4
    assert pycliques.verify_induced_octahedron(
        graph, result.antipodal_pairs, require_dominating=True
    )
    assert result.best_common_neighbors == 0


def test_nondominating_target_is_not_accepted_as_retract() -> None:
    graph = nx.complete_multipartite_graph(2, 2, 2)
    graph.add_edges_from((6, v) for v in range(6))

    inconclusive = pycliques.find_induced_octahedron_compatibility(
        graph, target_dimension=3, max_restarts=12, local_steps=12
    )
    assert inconclusive.status is pycliques.OctahedronSearchStatus.LIMIT_REACHED
    assert inconclusive.best_dimension >= 3
    assert inconclusive.antipodal_pairs is None

    ordinary = pycliques.find_induced_octahedron_compatibility(
        graph, target_dimension=3, max_restarts=12, require_dominating=False
    )
    assert ordinary.status is pycliques.OctahedronSearchStatus.FOUND
    assert ordinary.antipodal_pairs is not None
    assert not pycliques.verify_induced_octahedron(
        graph, ordinary.antipodal_pairs, require_dominating=True
    )


def test_impossible_dimension_is_inconclusive() -> None:
    result = pycliques.find_induced_octahedron_compatibility(
        nx.octahedral_graph(), target_dimension=4
    )
    assert result.status is pycliques.OctahedronSearchStatus.LIMIT_REACHED
    assert result.restarts == 0


@pytest.mark.parametrize(
    "options",
    [
        {"target_dimension": 2},
        {"target_dimension": True},
        {"target_dimension": 3, "max_restarts": 0},
        {"target_dimension": 3, "local_steps": -1},
        {"target_dimension": 3, "pair_samples": 0},
        {"target_dimension": 3, "seed": True},
    ],
)
def test_invalid_parameters(options: dict) -> None:
    with pytest.raises(ValueError):
        pycliques.find_induced_octahedron_compatibility(
            nx.octahedral_graph(), **options
        )


@pytest.mark.parametrize(
    "bad_graph",
    [nx.DiGraph(), nx.MultiGraph(), nx.Graph([(0, 0)])],
)
def test_invalid_graphs(bad_graph: nx.Graph) -> None:
    with pytest.raises(ValueError, match="undirected loopless"):
        pycliques.find_induced_octahedron_compatibility(
            bad_graph, target_dimension=3
        )


def _canonical_clique_graph(graph: nx.Graph) -> nx.Graph:
    result = pycliques.clique_graph(graph, bound=1200)
    assert result is not None
    order = sorted(
        result, key=lambda clique: sum(1 << int(v) for v in clique)
    )
    return nx.relabel_nodes(result, {clique: i for i, clique in enumerate(order)})


def test_snub_o20_discovered_without_stored_coordinates() -> None:
    """Scientific benchmark: no named snub helper or witness is imported."""
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from((i, i + 1) for i in range(3))
    graph.add_edges_from((i, i + 1) for i in range(4, 7))
    graph.add_edges_from(
        (i, 4 + j)
        for i in range(4)
        for j in range(4)
        if i in (0, 3) or j in (0, 3)
    )
    for _ in range(4):
        graph = _canonical_clique_graph(graph)
    assert len(graph) == 1076

    result = pycliques.find_induced_octahedron_compatibility(
        graph,
        target_dimension=20,
        max_restarts=500,
        seed=24002,
        require_dominating=True,
    )
    assert result.status is pycliques.OctahedronSearchStatus.FOUND
    assert result.antipodal_pairs is not None
    assert len(result.antipodal_pairs) >= 20
    assert pycliques.verify_induced_octahedron(
        graph, result.antipodal_pairs, require_dominating=True
    )
