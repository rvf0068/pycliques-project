"""General finite target-family retractions, positive/negative/bounded cases."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest
from pycliques.named import (
    complement_of_cycle,
    octahedron,
    snub_disphenoid,
    suspension_of_cycle,
)
from pycliques.retractions import BoundedRetractionResult, RetractionSearchStatus
from pycliques.targeted_retractions import (
    TargetRetractionResult,
    TargetSearchStatus,
    search_targeted_retractions,
    verify_targeted_retraction,
)


@pytest.mark.parametrize(
    "family,graph",
    [
        ("octahedra", octahedron(3)),
        ("octahedra", octahedron(4)),
        ("suspensions", suspension_of_cycle(5)),
        ("suspensions", suspension_of_cycle(6)),
        ("cycle_complements", complement_of_cycle(8)),
        ("cycle_complements", complement_of_cycle(10)),
    ],
)
def test_positive_retraction_on_target_itself(family, graph):
    found = search_targeted_retractions(
        graph,
        families=(family,),
        max_target_order=20,
    )[family]
    assert found.status is TargetSearchStatus.FOUND
    assert found.projection is not None
    assert verify_targeted_retraction(graph, found)


def test_retraction_with_nonneighbor_vertex_identification():
    graph = octahedron(3)
    graph.add_node(6)
    graph.add_edges_from((6, v) for v in octahedron(3)[0])
    found = search_targeted_retractions(graph, families=("octahedra",))
    assert found["octahedra"].status is TargetSearchStatus.FOUND
    assert verify_targeted_retraction(graph, found["octahedra"])


def test_exact_homology_exclusions_and_unknown_degree_semantics():
    graph = octahedron(3)
    excluded = search_targeted_retractions(
        graph,
        families=("octahedra",),
        betti_numbers={2: 0},
    )["octahedra"]
    assert excluded.status is TargetSearchStatus.HOMOLOGY_EXCLUDED
    unknown = search_targeted_retractions(
        graph,
        families=("octahedra",),
        betti_numbers={3: 0},
    )["octahedra"]
    assert unknown.status is TargetSearchStatus.FOUND
    assert (
        search_targeted_retractions(
            graph,
            families=("octahedra",),
            betti_numbers=None,
        )["octahedra"].status
        is TargetSearchStatus.FOUND
    )


def test_s2_homology_excludes_o4_but_keeps_o3():
    graph = octahedron(3)
    result = search_targeted_retractions(
        graph,
        families=("octahedra",),
        betti_numbers={0: 0, 1: 0, 2: 1, 3: 0},
        max_target_order=10,
    )
    assert result["octahedra"].status is TargetSearchStatus.FOUND
    assert result["octahedra"].target_kind == "octahedra:3"


def test_negative_exact_exhaustion_on_small_graphs():
    graph = snub_disphenoid()
    results = search_targeted_retractions(graph, max_target_order=8)
    assert all(x.status is TargetSearchStatus.EXHAUSTED for x in results.values())
    assert not any(x.projection is not None for x in results.values())


def test_no_fitting_targets_not_homology_exclusion():
    graph = nx.cycle_graph(4)
    r = search_targeted_retractions(graph, families=("octahedra",))
    assert r["octahedra"].status is TargetSearchStatus.EXHAUSTED


def test_restricted_target_limit_not_confused_with_global_claim():
    graph = octahedron(4)
    r = search_targeted_retractions(
        graph,
        families=("octahedra",),
        max_target_order=5,
    )["octahedra"]
    assert r.status is TargetSearchStatus.EXHAUSTED


def test_pattern_budget_is_not_exhaustion():
    graph = nx.complete_graph(4)
    graph.add_node(4)
    graph.add_edges_from(((4, 0), (4, 1)))
    # A zero-graph target test does not hit a limit.
    result = search_targeted_retractions(
        graph,
        families=("octahedra",),
        max_patterns=1,
    )["octahedra"]
    assert result.status is TargetSearchStatus.EXHAUSTED


@pytest.mark.parametrize("betti", [{2: -1}, {-1: 1}, {2: 1.5}, {True: 0}, {2: True}])
def test_invalid_betti_metadata_rejected(betti):
    with pytest.raises(ValueError, match="Betti"):
        search_targeted_retractions(nx.cycle_graph(5), betti_numbers=betti)


def test_no_circular_homology_dependency():
    import pycliques.targeted_retractions as module

    assert "pycombtop" not in module.__dict__


def test_target_replay_rejects_tampering_and_false_family_claims():
    g = octahedron(3)
    original = search_targeted_retractions(g, families=("octahedra",))["octahedra"]
    assert verify_targeted_retraction(g, original)
    assert not verify_targeted_retraction(
        g, replace(original, status=TargetSearchStatus.EXHAUSTED)
    )
    bad_map = dict(original.projection)
    bad_map[0] = 9999
    assert not verify_targeted_retraction(g, replace(original, projection=bad_map))
    assert not verify_targeted_retraction(
        g, replace(original, target_kind="invalid_kind")
    )
    assert not verify_targeted_retraction(
        g, replace(original, target_kind="suspensions:5")
    )
    assert not verify_targeted_retraction(
        g, replace(original, target_kind="octahedra:4")
    )

    # O4 is not a suspension of a cycle, even though 8=6+2.
    h = octahedron(4)
    false_target = TargetRetractionResult(
        "suspensions",
        TargetSearchStatus.FOUND,
        target_kind="suspensions:6",
        target_vertices=tuple(h),
        projection={v: v for v in h},
    )
    assert not verify_targeted_retraction(h, false_target)


@pytest.mark.parametrize(
    "bad_graph",
    [
        nx.MultiGraph(),
        nx.DiGraph(),
        nx.Graph([(0, 0)]),
    ],
)
def test_invalid_host_graph_is_rejected(bad_graph):
    with pytest.raises(ValueError, match="simple loopless"):
        search_targeted_retractions(bad_graph)


@pytest.mark.parametrize(
    "family_names",
    [
        ("octahedra", "octahedra"),
        ("invalid_family",),
    ],
)
def test_invalid_target_family_names_are_rejected(family_names):
    with pytest.raises(ValueError, match="target family"):
        search_targeted_retractions(octahedron(3), families=family_names)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_patterns": 0},
        {"max_states": 0},
        {"max_assignments": 0},
        {"max_target_order": 0},
    ],
)
def test_invalid_search_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError, match="Search bounds"):
        search_targeted_retractions(octahedron(3), **kwargs)


def test_actual_pattern_and_csp_budget_limits():
    graph = octahedron(4)
    # O4 contains induced O3, but homology forbids a retraction onto O3.
    pattern_limited = search_targeted_retractions(
        graph, families=("octahedra",), max_patterns=1
    )["octahedra"]
    assert pattern_limited.status is TargetSearchStatus.LIMIT_REACHED
    assert pattern_limited.induced_targets_checked == 1

    csp_limited = search_targeted_retractions(
        graph, families=("octahedra",), max_states=1
    )["octahedra"]
    assert csp_limited.status is TargetSearchStatus.LIMIT_REACHED


def test_global_state_budget_applies_across_distinct_induced_targets(monkeypatch):
    import pycliques.targeted_retractions as search

    def exhausted_one_state(*_args, **_kwargs):
        return BoundedRetractionResult(
            RetractionSearchStatus.EXHAUSTED, None, None, 1, 0
        )

    monkeypatch.setattr(search, "bounded_retraction", exhausted_one_state)
    result = search_targeted_retractions(
        octahedron(4), families=("octahedra",), max_states=1
    )["octahedra"]
    assert result.status is TargetSearchStatus.LIMIT_REACHED
    assert result.induced_targets_checked == 2


def test_repeated_pattern_does_not_duplicate_target_csp(monkeypatch):
    import pycliques.targeted_retractions as search

    vertices = frozenset(range(6))

    def repeated(_graph, _allowed):
        yield 3, vertices
        yield 3, vertices

    def no_retraction(*_args, **_kwargs):
        return BoundedRetractionResult(
            RetractionSearchStatus.EXHAUSTED, None, None, 1, 0
        )

    monkeypatch.setitem(search._GENERATORS, "octahedra", repeated)
    monkeypatch.setattr(search, "bounded_retraction", no_retraction)
    result = search_targeted_retractions(octahedron(4), families=("octahedra",))[
        "octahedra"
    ]
    assert result.status is TargetSearchStatus.EXHAUSTED
    assert result.patterns_checked == 2
    assert result.induced_targets_checked == 1


def test_positive_witness_always_verified_before_return(monkeypatch):
    import pycliques.targeted_retractions as search

    monkeypatch.setattr(search, "verify_targeted_retraction", lambda *_: False)
    with pytest.raises(AssertionError, match="witness failed replay"):
        search_targeted_retractions(octahedron(3), families=("octahedra",))
