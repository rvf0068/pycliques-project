"""Small exhaustive fixtures for selected Helly edge-cover search."""

import networkx as nx
from pycliques import (
    HellyCoverStatus,
    HellyCoverVerificationStatus,
    prune_helly_cover_candidates,
)

from research.selected_helly_cover import (
    SelectedHellyCoverStatus,
    search_selected_helly_cover,
)


def test_search_finds_and_independently_verifies_a_selected_cover() -> None:
    graph = nx.path_graph(3)
    candidates = (frozenset({0, 1}), frozenset({1, 2}))
    pruned = prune_helly_cover_candidates(graph, candidates)

    assert pruned.status is HellyCoverStatus.PRUNED_EXHAUSTIVE
    result = search_selected_helly_cover(graph, pruned)

    assert result.status is SelectedHellyCoverStatus.FOUND_VERIFIED_COVER
    assert result.selected_family is not None
    assert result.verification is not None
    assert result.verification.status is HellyCoverVerificationStatus.VERIFIED
    assert set(result.selected_family) == set(candidates)
    assert result.to_json() == result.to_json()


def test_search_exhausts_branches_for_required_triangle_cover() -> None:
    graph = nx.complete_graph(3)
    required = frozenset({0, 1})
    candidates = (
        required,
        frozenset({0, 2}),
        frozenset({1, 2}),
    )
    pruned = prune_helly_cover_candidates(
        graph,
        candidates,
        required_facets=(required,),
        private_edges={required: required},
    )

    result = search_selected_helly_cover(graph, pruned)

    assert result.status is SelectedHellyCoverStatus.NO_COVER_CERTIFIED
    assert result.states_explored > 0
    assert result.selected_family is None
    assert result.verification is None
    assert result.pruning_result is pruned
    assert "surviving candidate universe" in result.diagnostics


def test_state_and_depth_budgets_are_inconclusive() -> None:
    graph = nx.path_graph(3)
    candidates = (frozenset({0, 1}), frozenset({1, 2}))
    pruned = prune_helly_cover_candidates(graph, candidates)

    state_limited = search_selected_helly_cover(graph, pruned, max_states=0)
    depth_limited = search_selected_helly_cover(graph, pruned, max_depth=1)

    assert state_limited.status is SelectedHellyCoverStatus.INCONCLUSIVE_BUDGET
    assert state_limited.states_explored == 0
    assert depth_limited.status is SelectedHellyCoverStatus.INCONCLUSIVE_BUDGET


def test_inconclusive_pruning_never_becomes_a_negative_search_result() -> None:
    graph = nx.complete_graph(3)
    candidates = tuple(frozenset(edge) for edge in ((0, 1), (0, 2), (1, 2)))
    pruned = prune_helly_cover_candidates(
        graph,
        candidates,
        max_support_checks=0,
    )

    assert pruned.status is HellyCoverStatus.INCONCLUSIVE_BUDGET
    result = search_selected_helly_cover(graph, pruned)
    assert result.status is SelectedHellyCoverStatus.INCONCLUSIVE_BUDGET
    assert result.states_explored == 0
