"""Tests for graph-independent induced-octahedron amplification."""

import networkx as nx
import pycliques
import pytest


def test_find_o3_without_a_saved_embedding() -> None:
    graph = nx.octahedral_graph()
    result = pycliques.find_induced_octahedron_amplified(
        graph, target_dimension=3, max_restarts=20, seed=11
    )
    assert result.status is pycliques.OctahedronSearchStatus.FOUND
    assert result.antipodal_pairs is not None
    assert len(result.antipodal_pairs) == 3
    assert pycliques.verify_induced_octahedron(
        graph, result.antipodal_pairs, require_dominating=True
    )
    witness = pycliques.find_octahedral_retraction(
        graph, result.antipodal_pairs, max_states=100
    )
    assert witness.status is pycliques.RetractionSearchStatus.FOUND


def test_repair_from_a_smaller_seed_under_arbitrary_vertex_labels() -> None:
    original = nx.octahedral_graph()
    mapping = {v: ("vertex", v) for v in original}
    graph = nx.relabel_nodes(original, mapping)
    pairs = [(mapping[a], mapping[b]) for a, b in nx.non_edges(original)]
    result = pycliques.find_induced_octahedron_amplified(
        graph, target_dimension=3, seed_pairs=pairs[:2], max_restarts=50
    )
    assert result.status is pycliques.OctahedronSearchStatus.FOUND
    assert result.antipodal_pairs is not None
    assert pycliques.verify_induced_octahedron(
        graph, result.antipodal_pairs, require_dominating=True
    )


def test_induced_target_not_automatically_dominating() -> None:
    graph = nx.octahedral_graph()
    universal = 6
    graph.add_edges_from((universal, v) for v in range(6))
    pairs = list(nx.non_edges(graph))
    assert pycliques.verify_induced_octahedron(graph, pairs)
    assert not pycliques.verify_induced_octahedron(
        graph, pairs, require_dominating=True
    )
    result = pycliques.find_induced_octahedron_amplified(
        graph, target_dimension=3, max_restarts=20, require_dominating=False
    )
    assert result.status is pycliques.OctahedronSearchStatus.FOUND


def test_limit_is_not_negative_certificate() -> None:
    result = pycliques.find_induced_octahedron_amplified(
        nx.cycle_graph(7), target_dimension=4, max_restarts=5
    )
    assert result.status is pycliques.OctahedronSearchStatus.LIMIT_REACHED
    assert result.antipodal_pairs is None


@pytest.mark.parametrize(
    "options",
    [
        {"target_dimension": 2},
        {"target_dimension": 3, "max_restarts": 0},
        {"target_dimension": 3, "pair_samples": 0},
        {"target_dimension": 3, "exploration_restarts": -1},
        {"target_dimension": 3, "seed": True},
    ],
)
def test_invalid_search_parameters(options: dict) -> None:
    with pytest.raises(ValueError):
        pycliques.find_induced_octahedron_amplified(nx.octahedral_graph(), **options)


def test_invalid_embedding_is_rejected() -> None:
    graph = nx.octahedral_graph()
    with pytest.raises(ValueError, match="seed_pairs"):
        pycliques.find_induced_octahedron_amplified(
            graph, target_dimension=3, seed_pairs=[(0, 1)]
        )
    assert not pycliques.verify_induced_octahedron(graph, [(0, 1)])
    assert not pycliques.verify_induced_octahedron(graph, [(0, 0)])
    assert not pycliques.verify_induced_octahedron(graph, [(0, 9)])


def test_nondominating_search_remains_inconclusive() -> None:
    octahedron = nx.octahedral_graph()
    graph = octahedron.copy()
    graph.add_edges_from((6, v) for v in octahedron)
    seed_pairs = list(nx.non_edges(octahedron))[:2]
    result = pycliques.find_induced_octahedron_amplified(
        graph,
        target_dimension=3,
        seed_pairs=seed_pairs,
        max_restarts=120,
        exploration_restarts=2,
        seed=9,
    )
    assert result.status is pycliques.OctahedronSearchStatus.LIMIT_REACHED
    assert result.best_dimension == 3


def test_complete_graph_has_no_antipodal_extension() -> None:
    result = pycliques.find_induced_octahedron_amplified(
        nx.complete_graph(8),
        target_dimension=3,
        max_restarts=5,
        exploration_restarts=2,
    )
    assert result.status is pycliques.OctahedronSearchStatus.LIMIT_REACHED
    assert result.best_dimension == 0


def test_four_pairs_can_be_discovered_or_retained() -> None:
    graph = nx.complete_multipartite_graph(2, 2, 2, 2)
    found = pycliques.find_induced_octahedron_amplified(
        graph, target_dimension=4, max_restarts=50
    )
    assert found.status is pycliques.OctahedronSearchStatus.FOUND
    seed_pairs = list(nx.non_edges(graph))
    retained = pycliques.find_induced_octahedron_amplified(
        graph, target_dimension=3, seed_pairs=seed_pairs, max_restarts=100
    )
    assert retained.antipodal_pairs is not None
    assert len(retained.antipodal_pairs) == 4
    assert pycliques.verify_induced_octahedron(
        graph, retained.antipodal_pairs, require_dominating=True
    )


@pytest.mark.parametrize(
    "bad_graph",
    [
        nx.DiGraph(),
        nx.MultiGraph(),
        nx.Graph([(0, 0)]),
    ],
)
def test_invalid_graphs_are_rejected(bad_graph: nx.Graph) -> None:
    with pytest.raises(ValueError, match="simple undirected"):
        pycliques.find_induced_octahedron_amplified(
            bad_graph, target_dimension=3
        )


def test_missing_cross_edges_are_rejected() -> None:
    assert not pycliques.verify_induced_octahedron(
        nx.empty_graph(4), [(0, 1), (2, 3)]
    )
