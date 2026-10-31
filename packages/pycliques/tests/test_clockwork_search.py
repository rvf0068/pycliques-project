"""Tests for bounded induced-clockwork and retraction searches."""

import networkx as nx
from pycliques import (
    ClockworkSearchStatus,
    clockwork_graph,
    find_divergent_clockwork_retract,
    find_induced_clockwork_subgraphs,
    verify_clockwork_structure,
)
from pycliques.cliques import clique_graph
from pycliques.retractions import is_map, retracts
from pycliques.small import Verdict, classify_clique_behavior


def _motivating_graph() -> tuple[nx.Graph, nx.Graph, nx.Graph]:
    graph = nx.from_graph6_bytes(b"IEjdnpy^O")
    clique_graph_result = clique_graph(graph)
    assert clique_graph_result is not None
    cliques = sorted(
        (frozenset(clique) for clique in nx.find_cliques(graph)),
        key=lambda clique: tuple(sorted(clique)),
    )
    indices = [0, 2, 3, 4, 7, 8, 9, 10, 12]
    witness = clique_graph_result.subgraph([cliques[i] for i in indices]).copy()
    return graph, clique_graph_result, witness


def test_induced_search_reconstructs_motivating_clockwork() -> None:
    _graph, clique_graph_result, expected = _motivating_graph()

    result = find_induced_clockwork_subgraphs(
        clique_graph_result, max_order=9, max_candidates=1000
    )

    assert result.status is ClockworkSearchStatus.FOUND
    assert result.candidates_checked < result.candidates_total
    witness = result.witnesses[0]
    assert nx.is_isomorphic(witness.graph, expected)
    assert witness.graph.number_of_nodes() == 9
    assert witness.graph.number_of_edges() == 23
    assert witness.structure.s == 3
    assert witness.structure.good_segments
    assert witness.divergent is True
    valid, _ = verify_clockwork_structure(
        witness.graph,
        witness.structure.B_segments,
        witness.structure.C_segments,
        witness.structure.core_orders,
    )
    assert valid


def test_induced_search_is_invariant_under_relabeling() -> None:
    _graph, clique_graph_result, _expected = _motivating_graph()
    relabeling = {
        vertex: f"vertex-{index}"
        for index, vertex in enumerate(clique_graph_result.nodes())
    }
    relabeled = nx.relabel_nodes(clique_graph_result, relabeling)

    result = find_induced_clockwork_subgraphs(
        relabeled, max_order=9, max_candidates=1000
    )

    assert result.status is ClockworkSearchStatus.FOUND
    witness = result.witnesses[0]
    assert set(witness.vertices) <= set(relabeled)
    assert witness.structure.good_segments


def test_induced_search_distinguishes_exhaustion_and_limits() -> None:
    empty_search = find_induced_clockwork_subgraphs(nx.path_graph(8), max_order=8)
    assert empty_search.status is ClockworkSearchStatus.EXHAUSTED
    assert not empty_search.witnesses

    _graph, clique_graph_result, _expected = _motivating_graph()
    limited_search = find_induced_clockwork_subgraphs(
        clique_graph_result, max_order=9, max_candidates=1
    )
    assert limited_search.status is ClockworkSearchStatus.LIMIT_REACHED
    assert limited_search.candidates_checked == 1


def test_divergent_clockwork_retraction_certificates_are_verified() -> None:
    target = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    larger = target.copy()
    larger.add_edge("leaf", 0)

    for graph in (target, larger):
        result = find_divergent_clockwork_retract(
            graph,
            max_target_order=9,
            max_candidates=20,
            max_retraction_attempts=5,
        )
        certificate = result.certificate
        assert result.status is ClockworkSearchStatus.FOUND
        assert certificate is not None
        assert set(certificate.retraction_map) == set(graph)
        assert is_map(graph, certificate.target_graph, certificate.retraction_map)
        assert all(
            certificate.retraction_map[vertex] == vertex
            for vertex in certificate.target_vertices
        )
        induced_target = graph.subgraph(certificate.target_vertices)
        assert set(certificate.target_graph) == set(induced_target)
        assert {frozenset(edge) for edge in certificate.target_graph.edges()} == {
            frozenset(edge) for edge in induced_target.edges()
        }
        assert nx.is_isomorphic(certificate.target_graph, target)
        assert certificate.structure.good_segments


def test_classifier_uses_opt_in_retraction_stage() -> None:
    graph = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])

    behavior = classify_clique_behavior(
        graph,
        tries=1,
        bound=1,
        clockwork_depth=0,
        clockwork_retract_depth=1,
        clockwork_max_order=9,
        clockwork_max_candidates=20,
    )

    assert behavior.verdict is Verdict.DIVERGENT
    assert behavior.certificate is not None
    assert behavior.certificate.rule == "induced_clockwork_retraction"
    assert behavior.certificate.iterate_index == 0


def test_motivating_containment_does_not_certify_divergence() -> None:
    graph, clique_graph_result, expected = _motivating_graph()
    assert graph.number_of_nodes() == 10
    assert graph.number_of_edges() == 27
    assert retracts(clique_graph_result, expected) is False

    retraction_search = find_divergent_clockwork_retract(
        clique_graph_result,
        max_target_order=9,
        max_candidates=715,
        max_retraction_attempts=30,
    )
    assert retraction_search.status is ClockworkSearchStatus.EXHAUSTED
    assert retraction_search.certificate is None

    behavior = classify_clique_behavior(
        graph,
        tries=1,
        bound=20,
        induced_clockwork_depth=2,
        clockwork_max_order=9,
        clockwork_max_candidates=1000,
    )

    assert behavior.verdict is Verdict.INDETERMINATE
    assert behavior.observations
    assert behavior.observations[0].iterate_index == 1
    assert behavior.certificate is None
    assert "containment alone does not certify divergence" in behavior.reason
