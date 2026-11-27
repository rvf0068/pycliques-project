"""Tests for simplicial maps, contiguity, and induced chain maps."""

import pytest
from pycombtop import (
    SimplicialComplex,
    SimplicialMap,
    are_contiguous,
    induced_chain_map,
    prism_chain_homotopy,
    simplicial_boundary_matrix,
    simplicial_chain_basis,
)
from sympy import eye, zeros


def test_simplicial_map_can_collapse_simplices():
    source = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    target = SimplicialComplex({"a", "b"}, [{"a", "b"}])
    mapping = SimplicialMap(source, target, {0: "a", 1: "a", 2: "b"})

    assert mapping(0) == "a"
    assert mapping.image_simplex({0, 1, 2}) == {"a", "b"}
    assert mapping.image_simplex({0, 1}) == {"a"}


def test_simplicial_map_checks_exact_vertex_domain_and_target_faces():
    source = SimplicialComplex({0, 1, "ghost"}, [{0, 1}])
    target = SimplicialComplex({"a"}, [{"a"}])

    assert SimplicialMap(source, target, {0: "a", 1: "a"}).vertex_map == {
        0: "a",
        1: "a",
    }
    with pytest.raises(ValueError, match="exactly source.vertex_set"):
        SimplicialMap(source, target, {0: "a", 1: "a", "ghost": "a"})
    with pytest.raises(ValueError, match="target.vertex_set"):
        SimplicialMap(source, target, {0: "a", 1: "missing"})
    with pytest.raises(ValueError, match="target face"):
        SimplicialMap(
            SimplicialComplex({0, 1}, [{0, 1}]),
            SimplicialComplex({"a", "b"}, [{"a"}, {"b"}]),
            {0: "a", 1: "b"},
        )


def test_simplicial_map_rejects_nonfaces_as_simplex_arguments():
    complex_ = SimplicialComplex({0, 1, 2}, [{0, 1}, {1, 2}])
    mapping = SimplicialMap(complex_, complex_, {0: 0, 1: 1, 2: 2})

    with pytest.raises(ValueError, match="simplex of the source"):
        mapping.image_simplex({0, 2})
    with pytest.raises(ValueError, match="simplex of the source"):
        mapping.image_simplex({"ghost"})


def test_simplicial_map_handles_void_and_empty_face_complexes():
    void = SimplicialComplex(set(), [])
    unit = SimplicialComplex(set(), [set()])

    assert SimplicialMap(void, void, {}) == SimplicialMap(void, void, {})
    assert SimplicialMap(unit, unit, {}).image_simplex(set()) == set()
    with pytest.raises(ValueError, match="target face"):
        SimplicialMap(unit, void, {})


def test_contiguity_checks_maximal_source_simplices():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = SimplicialComplex({"a", "b", "c"}, [{"a", "b", "c"}])
    first = SimplicialMap(source, target, {0: "a", 1: "b"})
    second = SimplicialMap(source, target, {0: "b", 1: "c"})

    assert are_contiguous(first, second)
    assert first.is_contiguous_to(second)
    assert second.is_contiguous_to(first)


def test_contiguity_can_fail_even_when_both_maps_are_simplicial():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = SimplicialComplex(
        {"a", "b", "c"}, [{"a", "b"}, {"b", "c"}, {"a", "c"}]
    )
    first = SimplicialMap(source, target, {0: "a", 1: "b"})
    second = SimplicialMap(source, target, {0: "b", 1: "c"})

    assert not are_contiguous(first, second)


def test_contiguity_requires_the_same_source_and_target():
    source = SimplicialComplex({0}, [{0}])
    other_source = SimplicialComplex({1}, [{1}])
    target = SimplicialComplex({"a"}, [{"a"}])
    first = SimplicialMap(source, target, {0: "a"})
    second = SimplicialMap(other_source, target, {1: "a"})

    with pytest.raises(ValueError, match="same source and target"):
        are_contiguous(first, second)


def test_chain_basis_respects_explicit_vertex_order_and_ignores_ghosts():
    complex_ = SimplicialComplex({0, 1, 2, "ghost"}, [{0, 1, 2}])

    assert simplicial_chain_basis(complex_, 0, vertex_order=[2, 0, 1]) == (
        frozenset({2}),
        frozenset({0}),
        frozenset({1}),
    )
    assert simplicial_chain_basis(complex_, 1, vertex_order=[2, 0, 1]) == (
        frozenset({2, 0}),
        frozenset({2, 1}),
        frozenset({0, 1}),
    )
    with pytest.raises(ValueError, match="exactly the actual vertices"):
        simplicial_chain_basis(complex_, 0, vertex_order=[0, 1, "ghost"])


def test_default_chain_basis_handles_mixed_hashable_labels():
    complex_ = SimplicialComplex({1, "a", (2, 3)}, [{1, "a", (2, 3)}])

    basis = simplicial_chain_basis(complex_, 0)
    assert len(basis) == 3
    assert set(basis) == {frozenset({1}), frozenset({"a"}), frozenset({(2, 3)})}


def test_chain_matrix_functions_accept_one_shot_vertex_orders():
    complex_ = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    boundary = simplicial_boundary_matrix(complex_, 1, vertex_order=iter([0, 1, 2]))
    assert boundary.shape == (3, 3)

    identity = SimplicialMap(complex_, complex_, {0: 0, 1: 1, 2: 2})
    matrix = induced_chain_map(
        identity,
        1,
        source_order=iter([0, 1, 2]),
        target_order=iter([0, 1, 2]),
    )
    assert matrix == eye(3)


def test_boundary_matrices_square_to_zero():
    complex_ = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])

    boundary_1 = simplicial_boundary_matrix(complex_, 1, vertex_order=[0, 1, 2])
    boundary_2 = simplicial_boundary_matrix(complex_, 2, vertex_order=[0, 1, 2])
    assert boundary_1 * boundary_2 == zeros(3, 1)
    assert simplicial_boundary_matrix(complex_, 0).shape == (0, 3)


def test_induced_chain_map_has_orientation_sign_and_collapses_degenerate_faces():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = SimplicialComplex({"a", "b"}, [{"a", "b"}])
    reverse = SimplicialMap(source, target, {0: "b", 1: "a"})

    matrix = induced_chain_map(
        reverse, 1, source_order=[0, 1], target_order=["a", "b"]
    )
    assert matrix == -eye(1)

    collapse = SimplicialMap(source, target, {0: "a", 1: "a"})
    assert induced_chain_map(collapse, 1).shape == (1, 1)
    assert induced_chain_map(collapse, 1) == zeros(1, 1)


def test_induced_chain_maps_commute_with_boundary():
    source = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    target = SimplicialComplex({"a", "b", "c"}, [{"a", "b", "c"}])
    mapping = SimplicialMap(source, target, {0: "c", 1: "a", 2: "b"})
    source_order = [0, 1, 2]
    target_order = ["a", "b", "c"]

    for degree in (1, 2):
        left = simplicial_boundary_matrix(
            target, degree, vertex_order=target_order
        ) * induced_chain_map(
            mapping,
            degree,
            source_order=source_order,
            target_order=target_order,
        )
        right = induced_chain_map(
            mapping,
            degree - 1,
            source_order=source_order,
            target_order=target_order,
        ) * simplicial_boundary_matrix(
            source, degree, vertex_order=source_order
        )
        assert left == right


def test_prism_operator_is_chain_homotopy_for_contiguous_maps():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = SimplicialComplex({"a", "b", "c"}, [{"a", "b", "c"}])
    first = SimplicialMap(source, target, {0: "a", 1: "b"})
    second = SimplicialMap(source, target, {0: "b", 1: "c"})
    source_order = [0, 1]
    target_order = ["a", "b", "c"]

    prism_0 = prism_chain_homotopy(
        first, second, 0, source_order=source_order, target_order=target_order
    )
    prism_1 = prism_chain_homotopy(
        first, second, 1, source_order=source_order, target_order=target_order
    )
    difference_0 = induced_chain_map(
        second, 0, source_order=source_order, target_order=target_order
    ) - induced_chain_map(
        first, 0, source_order=source_order, target_order=target_order
    )
    difference_1 = induced_chain_map(
        second, 1, source_order=source_order, target_order=target_order
    ) - induced_chain_map(
        first, 1, source_order=source_order, target_order=target_order
    )

    assert (
        simplicial_boundary_matrix(target, 1, vertex_order=target_order) * prism_0
        == difference_0
    )
    assert (
        simplicial_boundary_matrix(target, 2, vertex_order=target_order) * prism_1
        + prism_0
        * simplicial_boundary_matrix(source, 1, vertex_order=source_order)
        == difference_1
    )


def test_prism_operator_rejects_noncontiguous_maps():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = SimplicialComplex(
        {"a", "b", "c"}, [{"a", "b"}, {"b", "c"}, {"a", "c"}]
    )
    first = SimplicialMap(source, target, {0: "a", 1: "b"})
    second = SimplicialMap(source, target, {0: "b", 1: "c"})

    with pytest.raises(ValueError, match="requires contiguous maps"):
        prism_chain_homotopy(first, second, 0)
