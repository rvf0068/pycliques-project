from itertools import combinations, permutations

import networkx as nx
import pytest
from pycliques import (
    build_octahedral_boolean_model,
    degree_histogram,
    enumerate_maximal_intersecting_families,
    homotopy_clique_graph,
)
from pycliques.cliques import clique_graph


def _octahedron(n):
    matching = nx.disjoint_union_all([nx.complete_graph(2) for _ in range(n)])
    return nx.complement(matching)


def _partial_assignment(maximal_clique_of_h):
    support = 0
    assignment = 0
    for vertex, maximal_clique in maximal_clique_of_h:
        coordinate, sign = divmod(vertex, 2)
        support |= 1 << coordinate
        if sign:
            assignment |= 1 << coordinate
        assert vertex in maximal_clique
    return support, assignment


def _model_vertex_from_maximal_clique_of_y(maximal_clique_of_y, model):
    support_family = 0
    sign_word = 0
    covered_coordinates = 0
    for maximal_clique_of_h in maximal_clique_of_y:
        support, assignment = _partial_assignment(maximal_clique_of_h)
        assert not (covered_coordinates & support & (sign_word ^ assignment))
        covered_coordinates |= support
        sign_word = (sign_word & ~support) | assignment
        support_family |= 1 << support

    assert covered_coordinates == model.full_mask
    try:
        family_index = model.families.index(support_family)
    except ValueError as error:
        raise AssertionError("direct maximal clique has no modeled family") from error
    return sign_word, family_index


@pytest.mark.parametrize(
    ("n", "family_count", "containment_counts", "order", "size"),
    [
        (2, 2, (0, 1, 2), 8, 8),
        (3, 4, (0, 1, 3, 4), 32, 168),
        (4, 12, (0, 1, 6, 11, 12), 192, 6688),
        (5, 81, (0, 1, 20, 61, 80, 81), 2592, 1_275_120),
        (6, 2646, (0, 1, 168, 1323, 2478, 2645, 2646), 169_344, 5_648_384_832),
    ],
)
def test_octahedral_boolean_model_exact_counts(
    n,
    family_count,
    containment_counts,
    order,
    size,
):
    result = build_octahedral_boolean_model(n)

    assert result.is_exact
    assert result.model is not None
    assert result.model.family_count == family_count
    assert result.model.containment_counts == containment_counts
    assert result.model.order == order
    assert result.model.size == size


def test_family_enumeration_limits_are_explicitly_inconclusive():
    limited_by_families = enumerate_maximal_intersecting_families(
        4,
        max_families=2,
    )
    limited_by_subset_vertices = enumerate_maximal_intersecting_families(
        4,
        max_subset_vertices=14,
    )
    model_result = build_octahedral_boolean_model(4, max_families=2)

    assert limited_by_families.is_exact is False
    assert len(limited_by_families.families) == 2
    assert limited_by_subset_vertices.is_exact is False
    assert model_result.is_exact is False
    assert model_result.model is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_families": -1},
        {"max_subset_vertices": -1},
    ],
)
def test_family_enumeration_rejects_negative_limits(kwargs):
    with pytest.raises(ValueError):
        enumerate_maximal_intersecting_families(3, **kwargs)


@pytest.mark.parametrize("n", [0, -1])
def test_family_enumeration_requires_positive_dimension(n):
    with pytest.raises(ValueError):
        enumerate_maximal_intersecting_families(n)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_entire_adjacency_relation_matches_direct_iterated_clique_graph(n):
    model_result = build_octahedral_boolean_model(n)
    model = model_result.model
    assert model is not None

    h_graph = homotopy_clique_graph(_octahedron(n))
    first_clique_graph = clique_graph(h_graph)
    assert first_clique_graph is not None
    direct_graph = clique_graph(first_clique_graph)
    assert direct_graph is not None

    labeled_vertices = {
        vertex: _model_vertex_from_maximal_clique_of_y(vertex, model)
        for vertex in direct_graph
    }
    assert len(set(labeled_vertices.values())) == model.order
    assert set(labeled_vertices.values()) == set(model.vertices())
    for first, second in combinations(direct_graph, 2):
        assert direct_graph.has_edge(first, second) == model.adjacent(
            labeled_vertices[first],
            labeled_vertices[second],
        )


def test_vertex_degrees_match_implicit_neighbors_for_small_instances():
    for n in (2, 3, 4):
        model = build_octahedral_boolean_model(n).model
        assert model is not None
        for vertex in model.vertices():
            neighbors = list(model.neighbors(vertex))
            assert len(neighbors) == len(set(neighbors))
            assert model.degree(vertex) == len(neighbors)
            assert all(model.adjacent(vertex, neighbor) for neighbor in neighbors)


def test_n5_representative_cross_fiber_edges_and_degree_histogram():
    model = build_octahedral_boolean_model(5).model
    assert model is not None
    full = model.full_mask

    edge = next(
        (
            (0, first_family),
            (full ^ 1, second_family),
        )
        for first_family, first in enumerate(model.families)
        if first & (1 << 1)
        for second_family, second in enumerate(model.families)
        if second & (1 << 1)
    )
    assert model.adjacent(*edge)
    assert not model.adjacent(edge[0], (full, edge[1][1]))
    assert degree_histogram(model) == {
        847: 160,
        926: 160,
        967: 960,
        1008: 960,
        1049: 320,
        1090: 32,
    }


def test_coordinate_permutations_and_sign_flips_preserve_adjacency():
    model = build_octahedral_boolean_model(3).model
    assert model is not None
    vertices = list(model.vertices())
    pairs = list(combinations(vertices, 2))
    for permutation in permutations(range(model.n)):
        for sign_flip in range(1 << model.n):
            for first, second in pairs:
                transformed_first = model.transform_vertex(
                    first, permutation, sign_flip
                )
                transformed_second = model.transform_vertex(
                    second, permutation, sign_flip
                )
                assert model.adjacent(first, second) == model.adjacent(
                    transformed_first, transformed_second
                )


def test_vertex_transform_rejects_invalid_permutation_and_sign_word():
    model = build_octahedral_boolean_model(3).model
    assert model is not None

    with pytest.raises(ValueError):
        model.transform_vertex((0, 0), (0, 0, 2))
    with pytest.raises(ValueError):
        model.transform_vertex((0, 0), (0, 1, 2), 8)


def test_o6_counts_without_allocating_the_global_graph():
    model = build_octahedral_boolean_model(6).model

    assert model is not None
    assert not hasattr(model, "graph")
    assert model.order == 169_344
    assert model.size == 5_648_384_832
