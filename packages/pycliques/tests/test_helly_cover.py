from __future__ import annotations

from itertools import combinations

import networkx as nx
import pytest
from pycliques import (
    HellyCoverStatus,
    HellyCoverVerificationStatus,
    clique_graph,
    is_helly_family,
    prune_helly_cover_candidates,
    satisfies_helly_gate_condition,
    verify_helly_edge_cover,
)


def _maximal_completes(graph: nx.Graph) -> tuple[frozenset[int], ...]:
    return tuple(frozenset(clique) for clique in nx.find_cliques(graph))


def _all_completes(graph: nx.Graph) -> tuple[frozenset[int], ...]:
    vertices = tuple(graph.nodes())
    return tuple(
        frozenset(candidate)
        for size in range(1, len(vertices) + 1)
        for candidate in combinations(vertices, size)
        if all(
            graph.has_edge(first, second)
            for first, second in combinations(candidate, 2)
        )
    )


def _all_subfamilies(
    family: tuple[frozenset[int], ...],
):
    for mask in range(1 << len(family)):
        yield tuple(
            member for index, member in enumerate(family) if mask & (1 << index)
        )


def _halfcube_family(pair_count: int) -> tuple[frozenset[int], ...]:
    row_count = pair_count.bit_length()
    if pair_count != 1 << (row_count - 1):
        raise ValueError("pair_count must be a power of two greater than one")
    base_rows = [
        tuple([0] + [(word >> (column - 1)) & 1 for column in range(1, row_count)])
        for word in range(pair_count)
    ]
    rows = [
        tuple(base_rows[column][row] for column in range(pair_count))
        for row in range(row_count)
    ]
    rows.extend(tuple(1 - value for value in row) for row in tuple(rows))
    return tuple(
        frozenset(2 * column + row[column] for column in range(pair_count))
        for row in rows
    )


def test_gate_characterization_matches_direct_helly_test() -> None:
    points = range(3)
    possible_sets = tuple(
        frozenset(subset)
        for size in range(1, 4)
        for subset in combinations(points, size)
    )
    for family in _all_subfamilies(possible_sets):
        assert is_helly_family(family) == satisfies_helly_gate_condition(family)

    non_helly_triple = ({0, 1}, {1, 2}, {0, 2})
    assert not is_helly_family(non_helly_triple)
    assert not satisfies_helly_gate_condition(non_helly_triple)
    assert is_helly_family(())
    assert satisfies_helly_gate_condition(())
    assert is_helly_family(({0, 1},))
    assert satisfies_helly_gate_condition(({0, 1},))
    assert is_helly_family(({0, 1}, {0, 1}))
    assert satisfies_helly_gate_condition(({0, 1}, {0, 1}))
    assert satisfies_helly_gate_condition(({0, 1}, {2}))
    with pytest.raises(ValueError, match="nonempty"):
        is_helly_family((set(),))


@pytest.mark.parametrize(
    "graph",
    [
        nx.complete_graph(3),
        nx.cycle_graph(4),
        nx.octahedral_graph(),
    ],
)
def test_pruning_preserves_every_selected_facet_of_valid_covers(
    graph: nx.Graph,
) -> None:
    universe = (
        _maximal_completes(graph)
        if graph.number_of_nodes() > 4
        else _all_completes(graph)
    )
    pruned = prune_helly_cover_candidates(graph, universe)
    assert pruned.status is not HellyCoverStatus.INCONCLUSIVE_BUDGET
    surviving = set(pruned.surviving_completes)

    for family in _all_subfamilies(universe):
        checked = verify_helly_edge_cover(graph, family)
        if checked.status is HellyCoverVerificationStatus.VERIFIED:
            assert set(family) <= surviving


def test_local_support_accepts_gate_equal_to_covered_neighbor() -> None:
    graph = nx.path_graph(3)
    facet = frozenset({0, 1})
    second = frozenset({1, 2})
    result = prune_helly_cover_candidates(graph, (facet, second))
    assert result.status is HellyCoverStatus.PRUNED_EXHAUSTIVE
    assert facet in result.surviving_completes


def test_empty_boundary_is_vacuous() -> None:
    graph = nx.Graph()
    graph.add_nodes_from((0, 1, 2))
    graph.add_edge(0, 1)
    facet = frozenset({0, 1})
    result = prune_helly_cover_candidates(graph, (facet,))
    assert result.status is HellyCoverStatus.PRUNED_EXHAUSTIVE
    assert result.surviving_completes == (facet,)


def test_required_facet_removed_is_reported_as_normalized_exclusion() -> None:
    graph = nx.complete_graph(3)
    required = frozenset({0, 1})
    candidates = (required, frozenset({0, 2}), frozenset({1, 2}))
    result = prune_helly_cover_candidates(
        graph, candidates, required_facets=(required,)
    )
    assert result.status is HellyCoverStatus.NO_COVER_CERTIFIED
    assert result.required_facet_missing
    assert required not in result.surviving_completes


def test_uncovered_edge_certificate_is_sound_after_pruning() -> None:
    graph = nx.complete_graph(3)
    result = prune_helly_cover_candidates(graph, (frozenset({0, 1}), frozenset({1, 2})))
    assert result.status is HellyCoverStatus.NO_COVER_CERTIFIED
    assert result.uncovered_edge is not None
    assert not any(
        set(result.uncovered_edge) <= candidate
        for candidate in result.surviving_completes
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_candidates": 1},
        {"max_rounds": 0},
        {"max_support_checks": 0},
    ],
)
def test_budget_exhaustion_never_returns_negative_certificate(kwargs) -> None:
    graph = nx.complete_graph(3)
    candidates = tuple(
        frozenset(subset)
        for size in range(1, 4)
        for subset in combinations(range(3), size)
    )
    result = prune_helly_cover_candidates(graph, candidates, **kwargs)
    assert result.status is HellyCoverStatus.INCONCLUSIVE_BUDGET
    assert result.uncovered_edge is None
    assert not result.required_facet_missing


def test_validation_rejects_invalid_candidate_normalization() -> None:
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match="duplicate"):
        prune_helly_cover_candidates(graph, ({0, 1}, frozenset({0, 1})))
    with pytest.raises(ValueError, match="complete"):
        prune_helly_cover_candidates(graph, ({0, 2},))
    with pytest.raises(ValueError, match="candidate universe"):
        prune_helly_cover_candidates(graph, ({0, 1},), required_facets=({1, 2},))
    with pytest.raises(ValueError, match="private edge"):
        prune_helly_cover_candidates(
            graph,
            ({0, 1}, {1, 2}),
            required_facets=({0, 1},),
            private_edges={frozenset({0, 1}): frozenset({0, 2})},
        )
    with pytest.raises(ValueError, match="simple undirected"):
        prune_helly_cover_candidates(nx.MultiGraph(), ())
    graph_with_loop = nx.Graph()
    graph_with_loop.add_edge(0, 0)
    with pytest.raises(ValueError, match="self-loops"):
        prune_helly_cover_candidates(graph_with_loop, ({0},))
    with pytest.raises(ValueError, match="outside its facet"):
        prune_helly_cover_candidates(
            nx.complete_graph(3),
            ({0, 1},),
            required_facets=({0, 1},),
            fixed_gates={(frozenset({0, 1}), 0): 1},
        )
    with pytest.raises(ValueError, match="belong"):
        prune_helly_cover_candidates(
            nx.complete_graph(3),
            ({0, 1},),
            required_facets=({0, 1},),
            fixed_gates={(frozenset({0, 1}), 2): 2},
        )


def test_fixed_private_edge_and_gate_constraints_are_explicit() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from(((0, 1), (0, 2), (1, 2), (2, 3)))
    required = frozenset({0, 1, 2})
    private_edge = frozenset({0, 1})
    candidates = tuple(
        frozenset(clique)
        for size in range(1, 5)
        for clique in combinations(range(4), size)
        if all(
            graph.has_edge(first, second) for first, second in combinations(clique, 2)
        )
    )
    result = prune_helly_cover_candidates(
        graph,
        candidates,
        required_facets=(required,),
        private_edges={required: private_edge},
        fixed_gates={(required, 3): 2},
    )
    assert result.normalization_removed > 0
    assert required in result.surviving_completes
    assert all(
        not (3 in candidate and candidate & required and 2 not in candidate)
        for candidate in result.surviving_completes
    )
    assert all(
        candidate == required or not private_edge <= candidate
        for candidate in result.surviving_completes
    )
    serialized = result.to_dict()
    assert serialized["required_facets"] == [[0, 1, 2]]
    assert serialized["private_edges"] == [{"facet": [0, 1, 2], "edge": [0, 1]}]
    assert serialized["fixed_gates"] == [{"facet": [0, 1, 2], "point": 3, "gate": 2}]


def test_independent_verifier_returns_pairwise_intersecting_obstruction() -> None:
    graph = nx.complete_graph(3)
    family = ({0, 1}, {1, 2}, {0, 2})
    result = verify_helly_edge_cover(graph, family)
    assert result.status is HellyCoverVerificationStatus.INVALID
    assert result.uncovered_edge is None
    assert result.non_helly_subfamily is not None
    assert len(result.non_helly_subfamily) == 3


def test_verifier_accepts_empty_family_for_edgeless_graph() -> None:
    result = verify_helly_edge_cover(nx.empty_graph(2), ())
    assert result.status is HellyCoverVerificationStatus.VERIFIED


def test_arbitrary_hashable_labels_and_json_index_mapping() -> None:
    graph = nx.Graph()
    labels = ("left", ("tuple", 1), 42)
    graph.add_edges_from(((labels[0], labels[1]), (labels[1], labels[2])))
    candidate_labels = (
        frozenset((labels[0], labels[1])),
        frozenset((labels[1], labels[2])),
    )
    result = prune_helly_cover_candidates(graph, candidate_labels)
    assert result.status is HellyCoverStatus.PRUNED_EXHAUSTIVE
    assert result.index_to_label == labels
    serialized = result.to_dict()
    assert [item["index"] for item in serialized["index_to_label"]] == [0, 1, 2]
    assert result.to_json() == result.to_json()

    relabelled = nx.relabel_nodes(
        graph,
        {
            labels[0]: "a",
            labels[1]: ("b", 2),
            labels[2]: object(),
        },
        copy=True,
    )
    relabel = {
        labels[0]: "a",
        labels[1]: ("b", 2),
        labels[2]: next(vertex for vertex in relabelled if type(vertex) is object),
    }
    mapped_candidates = tuple(
        frozenset(relabel[vertex] for vertex in candidate)
        for candidate in candidate_labels
    )
    relabelled_result = prune_helly_cover_candidates(relabelled, mapped_candidates)
    assert relabelled_result.status is result.status
    assert set(relabelled_result.surviving_completes) == {
        frozenset(relabel[vertex] for vertex in candidate)
        for candidate in result.surviving_completes
    }


@pytest.mark.parametrize(("pair_count", "expected_facets"), [(4, 6), (8, 8)])
def test_canonical_halfcube_cover_and_real_clique_graph(
    pair_count: int, expected_facets: int
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(2 * pair_count))
    graph.add_edges_from(
        (first, second)
        for first in range(2 * pair_count)
        for second in range(first + 1, 2 * pair_count)
        if first // 2 != second // 2
    )
    family = _halfcube_family(pair_count)
    assert len(family) == expected_facets
    assert all(len(facet) == pair_count for facet in family)
    assert (
        verify_helly_edge_cover(graph, family).status
        is HellyCoverVerificationStatus.VERIFIED
    )

    pruned = prune_helly_cover_candidates(graph, family)
    assert pruned.status is HellyCoverStatus.PRUNED_EXHAUSTIVE
    assert set(pruned.surviving_completes) == set(family)

    intersection_graph = nx.Graph()
    intersection_graph.add_nodes_from(family)
    for first, second in combinations(family, 2):
        if first & second:
            intersection_graph.add_edge(first, second)
    actual_clique_graph = clique_graph(intersection_graph)
    assert actual_clique_graph is not None
    assert nx.is_isomorphic(actual_clique_graph, graph)
