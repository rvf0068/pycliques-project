"""Tests for bounded induced-clockwork and retraction searches."""

from unittest.mock import Mock

import networkx as nx
import pycliques.clockwork as clockwork_module
import pycliques.clockwork_search as search_module
import pycliques.retractions as retractions_module
import pycliques.small as small_module
import pytest
from pycliques import (
    CertificateVerificationStatus,
    CliqueBehavior,
    ClockworkRecognitionResult,
    ClockworkRecognitionStatus,
    ClockworkSearchStatus,
    clockwork_graph,
    find_divergent_clockwork_retract,
    find_induced_clockwork_subgraphs,
    recognize_clockwork_detailed,
    verify_clockwork_retraction_certificate,
    verify_clockwork_retraction_certificate_detailed,
    verify_clockwork_structure,
)
from pycliques.cliques import clique_graph
from pycliques.dominated import completely_pared_graph
from pycliques.retractions import (
    RetractionSearchStatus,
    bounded_retraction,
    is_map,
    retracts,
)
from pycliques.small import (
    CliqueSequence,
    Verdict,
    _test_clockwork,
    classify_clique_behavior,
    classify_clique_behavior_with_theorem_2_6,
)


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


def test_recognition_reports_fallback_incompleteness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    monkeypatch.setattr(
        clockwork_module,
        "_recognize_clockwork_fast",
        lambda _graph: (False, "fast recognition declined"),
    )

    limited = recognize_clockwork_detailed(
        target, fallback_max_order=9, fallback_max_candidates=1
    )

    assert limited.status is ClockworkRecognitionStatus.LIMIT_REACHED
    assert limited.candidates_checked == 1


def test_induced_search_preserves_incompleteness_with_a_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = nx.relabel_nodes(
        clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0]),
        {vertex: vertex + 1 for vertex in range(9)},
    )
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_nodes_from(target)
    graph.add_edges_from(target.edges())
    graph.add_edge(0, 1)
    recognize = search_module.recognize_clockwork_detailed

    def recognize_candidates(
        candidate: nx.Graph,
        *,
        fallback_max_order: int = 12,
        fallback_max_candidates: int = 5000,
    ) -> ClockworkRecognitionResult:
        if 0 in candidate:
            return ClockworkRecognitionResult(
                ClockworkRecognitionStatus.LIMIT_REACHED,
                None,
                "test recognition limit",
            )
        return recognize(
            candidate,
            fallback_max_order=fallback_max_order,
            fallback_max_candidates=fallback_max_candidates,
        )

    monkeypatch.setattr(
        search_module, "recognize_clockwork_detailed", recognize_candidates
    )

    result = find_induced_clockwork_subgraphs(graph, max_order=9, max_candidates=10)

    assert result.status is ClockworkSearchStatus.LIMIT_REACHED
    assert len(result.witnesses) == 1
    assert result.candidates_checked == result.candidates_total == 10


def test_bounded_retraction_handles_self_and_pendant_extensions() -> None:
    target = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    larger = target.copy()
    larger.add_edge("leaf", 0)

    self_result = bounded_retraction(target, target)
    pendant_result = bounded_retraction(larger, target)

    for result, source in ((self_result, target), (pendant_result, larger)):
        assert result.status is RetractionSearchStatus.FOUND
        assert result.retraction is not None
        assert is_map(source, target, result.retraction)
        assert all(result.retraction[vertex] == vertex for vertex in target)


@pytest.mark.parametrize(
    ("vertex_count", "target_vertices"),
    [(1100, [0]), (1002, [0, 1])],
)
def test_bounded_retraction_handles_deep_assignments(
    vertex_count: int, target_vertices: list[int]
) -> None:
    large = nx.empty_graph(vertex_count)
    target = large.subgraph(target_vertices).copy()

    result = bounded_retraction(
        large, target, max_states=vertex_count + 10, max_assignments=vertex_count + 10
    )

    assert result.status is RetractionSearchStatus.FOUND
    assert result.retraction is not None
    assert set(result.retraction) == set(large)
    assert set(result.retraction.values()) <= set(target)
    assert all(result.retraction[vertex] == vertex for vertex in target)
    assert is_map(large, target, result.retraction)


def test_bounded_retraction_exhaustion_and_limits() -> None:
    large = nx.wheel_graph(5)
    target = large.subgraph([1, 2, 3, 4]).copy()
    assignment_case = nx.complete_graph(3)
    assignment_target = assignment_case.subgraph([0, 1]).copy()

    exhausted = bounded_retraction(large, target)
    limited = bounded_retraction(assignment_case, assignment_target, max_states=1)

    assert exhausted.status is RetractionSearchStatus.EXHAUSTED
    assert limited.status is RetractionSearchStatus.LIMIT_REACHED
    assert limited.states_explored == 1
    assert limited.assignments_tried == 1


def test_bounded_retraction_does_not_enumerate_automorphisms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large = nx.complete_graph(8)
    target = large.subgraph([0, 1, 2]).copy()

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("bounded_retraction must not enumerate automorphisms")

    monkeypatch.setattr(retractions_module, "automorphisms", fail_if_called)

    result = bounded_retraction(large, target)

    assert result.status is RetractionSearchStatus.FOUND


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
        assert verify_clockwork_retraction_certificate(certificate)

    limited = find_divergent_clockwork_retract(
        larger,
        max_target_order=9,
        max_candidates=20,
        max_retraction_states=1,
    )
    assert limited.status is ClockworkSearchStatus.LIMIT_REACHED
    assert limited.states_explored == 1
    assert limited.assignments_tried == 1


def test_certificate_verifier_rejects_broken_iterate_provenance() -> None:
    target = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    result = find_divergent_clockwork_retract(target, max_target_order=9)
    assert result.certificate is not None
    certificate = result.certificate

    invalid = certificate.__class__(
        source_graph=certificate.source_graph,
        target_graph=certificate.target_graph,
        target_vertices=certificate.target_vertices,
        retraction_map=certificate.retraction_map,
        structure=certificate.structure,
        iterate_index=1,
        origin_graph=target,
        iterate_graphs=(target, target),
    )

    assert not verify_clockwork_retraction_certificate(invalid)


def test_retraction_search_validates_provenance_before_search() -> None:
    origin = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    next_iterate = clique_graph(origin)
    assert next_iterate is not None
    next_iterate = completely_pared_graph(next_iterate)

    with pytest.raises(ValueError, match="provenance|origin_graph|iterate_graphs"):
        find_divergent_clockwork_retract(origin, iterate_index=1)
    with pytest.raises(ValueError, match="length"):
        find_divergent_clockwork_retract(
            next_iterate,
            iterate_index=1,
            origin_graph=origin,
            iterate_graphs=(origin,),
        )
    wrong_origin = origin.copy()
    wrong_origin.add_node("extra")
    with pytest.raises(ValueError, match=r"iterate_graphs\[0\]"):
        find_divergent_clockwork_retract(
            next_iterate,
            iterate_index=1,
            origin_graph=wrong_origin,
            iterate_graphs=(origin, next_iterate),
        )
    with pytest.raises(ValueError, match="final iterate_graphs entry"):
        find_divergent_clockwork_retract(
            origin,
            iterate_index=1,
            origin_graph=origin,
            iterate_graphs=(origin, next_iterate),
        )

    valid_prefix_search = find_divergent_clockwork_retract(
        next_iterate,
        max_target_order=8,
        iterate_index=1,
        origin_graph=origin,
        iterate_graphs=(origin, next_iterate),
    )
    assert valid_prefix_search.status is ClockworkSearchStatus.EXHAUSTED


def test_detailed_replay_distinguishes_invalid_and_resource_limited() -> None:
    origin = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    source = clique_graph(origin)
    assert source is not None
    source = completely_pared_graph(source)
    recognition = recognize_clockwork_detailed(source)
    assert recognition.structure is not None
    certificate = search_module.ClockworkRetractionCertificate(
        source_graph=source,
        target_graph=source,
        target_vertices=frozenset(source),
        retraction_map={vertex: vertex for vertex in source},
        structure=recognition.structure,
        iterate_index=1,
        origin_graph=origin,
        iterate_graphs=(origin, source),
    )

    limited = verify_clockwork_retraction_certificate_detailed(
        certificate, clique_bound=1
    )
    verified = verify_clockwork_retraction_certificate_detailed(
        certificate, clique_bound=100
    )
    assert limited.status is CertificateVerificationStatus.LIMIT_REACHED
    assert verified.status is CertificateVerificationStatus.VERIFIED
    needed = sum(1 for _ in nx.find_cliques(origin))
    assert needed >= 1
    assert (
        verify_clockwork_retraction_certificate_detailed(
            certificate, clique_bound=needed
        ).status
        is CertificateVerificationStatus.VERIFIED
    )
    assert (
        verify_clockwork_retraction_certificate_detailed(
            certificate, clique_bound=needed - 1
        ).status
        is CertificateVerificationStatus.LIMIT_REACHED
    )

    changed_origin = origin.copy()
    changed_origin.add_node("tampered")
    tampered = certificate.__class__(
        source_graph=source,
        target_graph=source,
        target_vertices=frozenset(source),
        retraction_map={vertex: vertex for vertex in source},
        structure=recognition.structure,
        iterate_index=1,
        origin_graph=changed_origin,
        iterate_graphs=(changed_origin, source),
    )
    assert (
        verify_clockwork_retraction_certificate_detailed(
            tampered, clique_bound=100
        ).status
        is CertificateVerificationStatus.INVALID
    )

    edge_limited = verify_clockwork_retraction_certificate_detailed(
        certificate, clique_bound=100, max_edges=0
    )
    assert edge_limited.status is CertificateVerificationStatus.LIMIT_REACHED


def test_iterate_zero_certificate_verification_skips_clique_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    result = find_divergent_clockwork_retract(target, max_target_order=9)
    assert result.certificate is not None

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("iterate-zero verification must not construct K(G)")

    monkeypatch.setattr(search_module, "clique_graph", fail_if_called)

    verification = verify_clockwork_retraction_certificate_detailed(result.certificate)
    assert verification.status is CertificateVerificationStatus.VERIFIED


def test_classifier_uses_opt_in_retraction_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    retraction_search = Mock(wraps=small_module.find_divergent_clockwork_retract)
    monkeypatch.setattr(
        small_module, "find_divergent_clockwork_retract", retraction_search
    )

    behavior = classify_clique_behavior(
        graph,
        tries=1,
        bound=17,
        clockwork_depth=0,
        clockwork_retract_depth=1,
        clockwork_max_order=9,
        clockwork_max_candidates=20,
    )

    assert behavior.verdict is Verdict.DIVERGENT
    assert behavior.certificate is not None
    assert behavior.certificate.rule == "induced_clockwork_retraction"
    assert behavior.certificate.iterate_index == 0
    assert retraction_search.call_args is not None
    assert retraction_search.call_args.kwargs["replay_clique_bound"] == 17


def test_direct_clockwork_fallback_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    target = clockwork_graph([1, 1, 1], [[0], [0], [0]], 2, [1, 0])
    fallback_calls = 0
    original = clockwork_module._recognize_clockwork_fallback

    def counted_fallback(
        graph: nx.Graph, *, max_candidates: int
    ) -> clockwork_module.ClockworkRecognitionResult:
        nonlocal fallback_calls
        fallback_calls += 1
        return original(graph, max_candidates=max_candidates)

    monkeypatch.setattr(
        clockwork_module, "_recognize_clockwork_fast", lambda _graph: (False, "miss")
    )
    monkeypatch.setattr(
        clockwork_module, "_recognize_clockwork_fallback", counted_fallback
    )

    assert _test_clockwork(CliqueSequence(target), 1, 1) is None
    assert fallback_calls == 0

    _test_clockwork(
        CliqueSequence(target), 1, 1, fallback_enabled=True, fallback_max_order=9
    )
    assert fallback_calls > 0


def test_theorem_2_6_wrapper_forwards_clockwork_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.empty_graph(1)
    fast = CliqueBehavior(Verdict.INDETERMINATE, "stub", 1, False, graph)
    forwarded: dict[str, object] = {}

    def capture_options(_graph: nx.Graph, **options: object) -> CliqueBehavior:
        forwarded.update(options)
        return fast

    monkeypatch.setattr(small_module, "classify_clique_behavior", capture_options)
    monkeypatch.setattr(
        small_module, "classify_clockwork_pair_map", lambda *_args, **_kwargs: None
    )
    options = {
        "clockwork_depth": 1,
        "clockwork_fallback": True,
        "clockwork_fallback_max_order": 7,
        "clockwork_fallback_max_candidates": 23,
        "induced_clockwork_depth": 2,
        "clockwork_retract_depth": 3,
        "clockwork_max_order": 8,
        "clockwork_max_candidates": 29,
        "clockwork_max_retraction_attempts": 4,
        "clockwork_max_retraction_states": 31,
        "clockwork_max_retraction_assignments": 37,
    }

    result = classify_clique_behavior_with_theorem_2_6(
        graph,
        clockwork_depth=1,
        clockwork_fallback=True,
        clockwork_fallback_max_order=7,
        clockwork_fallback_max_candidates=23,
        induced_clockwork_depth=2,
        clockwork_retract_depth=3,
        clockwork_max_order=8,
        clockwork_max_candidates=29,
        clockwork_max_retraction_attempts=4,
        clockwork_max_retraction_states=31,
        clockwork_max_retraction_assignments=37,
    )

    assert result is fast
    assert {key: forwarded[key] for key in options} == options


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
