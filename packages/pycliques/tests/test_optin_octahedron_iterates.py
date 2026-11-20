"""Opt-in O_d retractions: defaults, budgets, and the nine-sphere witness."""

from __future__ import annotations

import networkx as nx
import pytest

import pycliques.small as small
from pycliques.retractions import verify_retraction_witness
from pycliques.targeted_retractions import (
    TargetSearchStatus,
    search_targeted_retractions,
    verify_targeted_retraction,
)


def test_heher_nine_sphere_third_iterate_is_certified_o4_divergent():
    """Reproduce McKay index 248687 using only generic library algorithms."""
    graph = nx.from_graph6_bytes(b"HEher^{")
    result = small.classify_clique_behavior(
        graph,
        tries=4,
        bound=80,
        clockwork_depth=0,
        retraction_depth=0,
        octahedron_retract_iterates=(3,),
        octahedron_dimensions=(4,),
    )
    assert result.verdict is small.Verdict.DIVERGENT
    assert result.certificate is not None
    cert = result.certificate
    assert cert.rule == "octahedron_retraction"
    assert cert.iterate_index == 3
    assert cert.target_label == "O_4"
    assert cert.target_status == "proven_divergent"
    assert cert.source is not None
    assert cert.target is not None
    assert cert.source.number_of_nodes() <= 60
    assert cert.target.number_of_nodes() == 8
    assert cert.map is not None
    assert verify_retraction_witness(cert.source, cert.target, cert.map)
    assert result.iterations_checked == 4
    assert result.bound_exceeded is False


def test_default_classifier_does_not_search_for_octahedral_retracts(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("Opt-in octahedral search unexpectedly ran by default")

    monkeypatch.setattr(small, "search_targeted_retractions", fail_if_called)
    result = small.classify_clique_behavior(nx.path_graph(4))
    assert result.verdict is small.Verdict.CONVERGENT


def test_bounded_target_search_only_selected_dimension():
    o4 = nx.complete_multipartite_graph(2, 2, 2, 2)
    selected = search_targeted_retractions(
        o4,
        families=("octahedra",),
        octahedron_dimensions=(4,),
        max_target_order=8,
    )["octahedra"]
    assert selected.status is TargetSearchStatus.FOUND
    assert selected.target_kind == "octahedra:4"
    assert verify_targeted_retraction(o4, selected)

    absent = search_targeted_retractions(
        o4,
        families=("octahedra",),
        octahedron_dimensions=(5,),
        max_target_order=10,
    )["octahedra"]
    assert absent.status is TargetSearchStatus.EXHAUSTED


def test_retraction_limits_remain_indeterminate():
    graph = nx.from_graph6_bytes(b"HEher^{")
    result = small.classify_clique_behavior(
        graph,
        tries=4,
        bound=80,
        clockwork_depth=0,
        retraction_depth=0,
        octahedron_retract_iterates=(3,),
        octahedron_dimensions=(4,),
        octahedron_max_patterns=1,
    )
    assert result.verdict is small.Verdict.INDETERMINATE
    assert "resource limit" in result.reason
    assert result.certificate is None


@pytest.mark.parametrize(
    "options",
    [
        {"octahedron_retract_iterates": (4,), "tries": 4},
        {"octahedron_retract_iterates": (-1,)},
        {"octahedron_retract_iterates": (1, 1)},
        {"octahedron_retract_iterates": (True,)},
        {"octahedron_dimensions": ()},
        {"octahedron_dimensions": (2,)},
        {"octahedron_dimensions": (4, 4)},
        {"octahedron_dimensions": (True,)},
        {"octahedron_max_patterns": 0},
        {"octahedron_max_states": 0},
        {"octahedron_max_assignments": 0},
    ],
)
def test_invalid_opt_in_options_rejected(options):
    with pytest.raises(ValueError, match="octahedron"):
        small.classify_clique_behavior(nx.cycle_graph(4), **options)


@pytest.mark.parametrize(
    "dimensions",
    [(), (2,), (4, 4), (True,)],
)
def test_invalid_target_dimension_selector_rejected(dimensions):
    with pytest.raises(ValueError, match="Octahedron dimensions"):
        search_targeted_retractions(
            nx.complete_graph(8),
            families=("octahedra",),
            octahedron_dimensions=dimensions,
        )
