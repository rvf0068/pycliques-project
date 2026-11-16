"""General finite target-family retractions, positive/negative/bounded cases."""

from __future__ import annotations

import networkx as nx
import pytest

from pycliques.named import (
    complement_of_cycle,
    octahedron,
    snub_disphenoid,
    suspension_of_cycle,
)
from pycliques.targeted_retractions import (
    TargetSearchStatus,
    search_targeted_retractions,
    verify_targeted_retraction,
)


@pytest.mark.parametrize("family,graph", [
    ("octahedra", octahedron(3)),
    ("octahedra", octahedron(4)),
    ("suspensions", suspension_of_cycle(5)),
    ("suspensions", suspension_of_cycle(6)),
    ("cycle_complements", complement_of_cycle(8)),
    ("cycle_complements", complement_of_cycle(10)),
])
def test_positive_retraction_on_target_itself(family, graph):
    found = search_targeted_retractions(
        graph, families=(family,), max_target_order=20,
    )[family]
    assert found.status is TargetSearchStatus.FOUND
    assert found.projection is not None
    assert verify_targeted_retraction(graph, found)


def test_retraction_with_nonneighbor_vertex_identification():
    graph = octahedron(3)
    graph.add_node(6)
    graph.add_edges_from((6,v) for v in octahedron(3)[0])
    found = search_targeted_retractions(graph, families=("octahedra",))
    assert found["octahedra"].status is TargetSearchStatus.FOUND
    assert verify_targeted_retraction(graph, found["octahedra"])


def test_exact_homology_exclusions_and_unknown_degree_semantics():
    graph = octahedron(3)
    excluded = search_targeted_retractions(
        graph, families=("octahedra",), betti_numbers={2:0},
    )["octahedra"]
    assert excluded.status is TargetSearchStatus.HOMOLOGY_EXCLUDED
    unknown = search_targeted_retractions(
        graph, families=("octahedra",), betti_numbers={3:0},
    )["octahedra"]
    assert unknown.status is TargetSearchStatus.FOUND
    assert search_targeted_retractions(
        graph, families=("octahedra",), betti_numbers=None,
    )["octahedra"].status is TargetSearchStatus.FOUND


def test_s2_homology_excludes_o4_but_keeps_o3():
    graph = octahedron(3)
    result = search_targeted_retractions(
        graph, families=("octahedra",),
        betti_numbers={0:0,1:0,2:1,3:0},max_target_order=10,
    )
    assert result["octahedra"].status is TargetSearchStatus.FOUND
    assert result["octahedra"].target_kind == "octahedra:3"


def test_negative_exact_exhaustion_on_small_graphs():
    graph = snub_disphenoid()
    results = search_targeted_retractions(graph,max_target_order=8)
    assert all(x.status is TargetSearchStatus.EXHAUSTED for x in results.values())
    assert not any(x.projection is not None for x in results.values())


def test_no_fitting_targets_not_homology_exclusion():
    graph = nx.cycle_graph(4)
    r = search_targeted_retractions(graph,families=("octahedra",))
    assert r["octahedra"].status is TargetSearchStatus.EXHAUSTED


def test_restricted_target_limit_not_confused_with_global_claim():
    graph = octahedron(4)
    r = search_targeted_retractions(
        graph,families=("octahedra",),max_target_order=5,
    )["octahedra"]
    assert r.status is TargetSearchStatus.EXHAUSTED


def test_pattern_budget_is_not_exhaustion():
    graph = nx.complete_graph(4)
    graph.add_node(4)
    graph.add_edges_from(((4,0),(4,1)))
    # A zero-graph target test does not hit a limit.
    result = search_targeted_retractions(
        graph, families=("octahedra",), max_patterns=1,
    )["octahedra"]
    assert result.status is TargetSearchStatus.EXHAUSTED


@pytest.mark.parametrize("betti",[{2:-1},{-1:1},{2:1.5},{True:0},{2:True}])
def test_invalid_betti_metadata_rejected(betti):
    with pytest.raises(ValueError, match="Betti"):
        search_targeted_retractions(nx.cycle_graph(5),betti_numbers=betti)


def test_no_circular_homology_dependency():
    import pycliques.targeted_retractions as module
    assert "pycombtop" not in module.__dict__
