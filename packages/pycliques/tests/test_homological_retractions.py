"""Test exact/one-sided bounds and relative iterate semantics."""

import networkx as nx
import pytest

from pycliques import (
    HomologicalRetractionObstruction,
    TargetSearchStatus,
    find_homological_retraction_obstruction,
    search_targeted_retractions,
)


def test_contractible_target_obstructed_only_in_later_iterate():
    host = {0: {5: 0}, 1: {5: 1}, 2: {20: 0}}
    target = {0: {5: 0}, 1: {5: 1}, 2: {20: 1}}
    assert find_homological_retraction_obstruction(host, target) == (
        HomologicalRetractionObstruction(2, 20, 0, 1)
    )
    assert find_homological_retraction_obstruction({0: host[0]}, target) is None


def test_unknown_is_not_zero_and_bounds_are_one_sided():
    assert find_homological_retraction_obstruction({1: {2: 0}}, {0: {2: 1}}) is None
    assert find_homological_retraction_obstruction({1: {2: 0}}, {1: {3: 1}}) is None
    assert find_homological_retraction_obstruction({0: {2: 3}}, {0: {2: 2}}) is None
    assert find_homological_retraction_obstruction({0: {2: 1}}, {0: {2: 2}}) == (
        HomologicalRetractionObstruction(0, 2, 1, 2)
    )


@pytest.mark.parametrize(
    "profile", [{-1: {0: 0}}, {0: {-1: 0}}, {0: {2: -1}}, {True: {2: 0}}]
)
def test_invalid_profile(profile):
    with pytest.raises(ValueError):
        find_homological_retraction_obstruction(profile, {})
    with pytest.raises(ValueError):
        find_homological_retraction_obstruction({}, profile)


def test_named_search_accepts_upper_bounds_and_exact_values():
    graph = nx.octahedral_graph()
    kwargs = dict(families=("octahedra",), octahedron_dimensions=(3,))
    excluded = search_targeted_retractions(graph, betti_upper_bounds={2: 0}, **kwargs)
    assert excluded["octahedra"].status is TargetSearchStatus.HOMOLOGY_EXCLUDED
    allowed = search_targeted_retractions(graph, betti_upper_bounds={2: 4}, **kwargs)
    assert allowed["octahedra"].status is TargetSearchStatus.FOUND
    with pytest.raises(ValueError, match="Betti"):
        search_targeted_retractions(
            graph, betti_numbers={2: 1}, betti_upper_bounds={2: 0}
        )
    with pytest.raises(ValueError, match="Betti"):
        search_targeted_retractions(graph, betti_upper_bounds={2: -1})


def test_positive_upper_bound_can_exclude_rank_two_target():
    graph = nx.complement(nx.cycle_graph(9))
    result = search_targeted_retractions(
        graph,
        families=("cycle_complements",),
        betti_upper_bounds={2: 1, 3: 0},
        max_target_order=9,
    )
    # C_8 complement has one S^2, so it is still admissible; C_9 requires two.
    assert result["cycle_complements"].status is not TargetSearchStatus.FOUND
