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
    target = SimplicialComplex({"a", "b", "c"}, [{"a", "b"}, {"b", "c"}, {"a", "c"}])
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

    matrix = induced_chain_map(reverse, 1, source_order=[0, 1], target_order=["a", "b"])
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
        ) * simplicial_boundary_matrix(source, degree, vertex_order=source_order)
        assert left == right


def test_prism_operator_is_chain_homotopy_for_contiguous_maps():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = SimplicialComplex({"a", "b", "c"}, [{"a", "b", "c"}])
    first = SimplicialMap(source, target, {0: "a", 1: "a"})
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
    assert prism_1 == eye(1)
    assert (
        simplicial_boundary_matrix(target, 2, vertex_order=target_order) * prism_1
        + prism_0 * simplicial_boundary_matrix(source, 1, vertex_order=source_order)
        == difference_1
    )


def test_prism_operator_rejects_noncontiguous_maps():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = SimplicialComplex({"a", "b", "c"}, [{"a", "b"}, {"b", "c"}, {"a", "c"}])
    first = SimplicialMap(source, target, {0: "a", 1: "b"})
    second = SimplicialMap(source, target, {0: "b", 1: "c"})

    with pytest.raises(ValueError, match="requires contiguous maps"):
        prism_chain_homotopy(first, second, 0)


def test_simplicial_map_argument_validation_equality_and_repr():
    from typing import Any, cast

    source = SimplicialComplex({0}, [{0}])
    target = SimplicialComplex({"a"}, [{"a"}])
    mapping = SimplicialMap(source, target, {0: "a"})

    with pytest.raises(TypeError, match="source and target"):
        SimplicialMap(cast(Any, object()), target, {})
    with pytest.raises(TypeError, match="source and target"):
        SimplicialMap(source, cast(Any, object()), {})
    with pytest.raises(TypeError, match="vertex_map"):
        SimplicialMap(source, target, cast(Any, []))

    assert mapping == SimplicialMap(source, target, {0: "a"})
    assert mapping != object()
    other_source = SimplicialComplex({1}, [{1}])
    assert mapping != SimplicialMap(other_source, target, {1: "a"})
    assert repr(mapping) == "SimplicialMap({0: 'a'})"


def test_contiguity_and_induced_map_reject_wrong_argument_types():
    from typing import Any, cast

    source = SimplicialComplex({0}, [{0}])
    target = SimplicialComplex({"a"}, [{"a"}])
    mapping = SimplicialMap(source, target, {0: "a"})

    with pytest.raises(TypeError, match="both arguments"):
        are_contiguous(cast(Any, object()), mapping)
    with pytest.raises(TypeError, match="both arguments"):
        are_contiguous(mapping, cast(Any, object()))
    with pytest.raises(TypeError, match="SimplicialMap instance"):
        induced_chain_map(cast(Any, object()), 0)


def test_chain_basis_validates_degrees_and_explicit_orders():
    complex_ = SimplicialComplex({0, 1}, [{0, 1}])

    for degree in (-1, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            simplicial_chain_basis(complex_, degree)

    with pytest.raises(ValueError, match="exactly once"):
        simplicial_chain_basis(complex_, 0, vertex_order=[0, 0])


def test_default_vertex_order_handles_nested_labels_and_rejects_key_collisions():
    nested_labels = {frozenset({1}), frozenset({2})}
    nested = SimplicialComplex(nested_labels, [nested_labels])
    assert len(simplicial_chain_basis(nested, 0)) == 2

    class ReprCollision:
        def __repr__(self) -> str:
            return "same"

    first, second = ReprCollision(), ReprCollision()
    ambiguous = SimplicialComplex({first, second}, [{first, second}])
    with pytest.raises(ValueError, match="ambiguous stable keys"):
        simplicial_chain_basis(ambiguous, 0)


def test_prism_operator_cancels_repeated_terms():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = SimplicialComplex({"a", "b", "c"}, [{"a", "b", "c"}])
    first = SimplicialMap(source, target, {0: "a", 1: "b"})
    second = SimplicialMap(source, target, {0: "b", 1: "c"})

    prism = prism_chain_homotopy(
        first,
        second,
        1,
        source_order=[0, 1],
        target_order=["a", "b", "c"],
    )
    assert prism == zeros(1, 1)


def test_simplicial_homotopy_equivalence_certificate_for_path_to_point():
    from pycombtop import (
        simplicial_homotopy_equivalence_certificate,
        verify_simplicial_homotopy_equivalence,
    )

    path = SimplicialComplex({0, 1, 2}, [{0, 1}, {1, 2}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, {0: "*", 1: "*", 2: "*"})
    backward = SimplicialMap(point, path, {"*": 1})

    certificate = simplicial_homotopy_equivalence_certificate(forward, backward)

    assert certificate is not None
    assert verify_simplicial_homotopy_equivalence(path, point, certificate)


def test_simplicial_homotopy_equivalence_accepts_multistep_contiguity():
    from pycombtop import simplicial_homotopy_equivalence_certificate

    path = SimplicialComplex({0, 1, 2, 3}, [{0, 1}, {1, 2}, {2, 3}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, dict.fromkeys(path.vertex_set, "*"))
    backward = SimplicialMap(point, path, {"*": 1})
    first = SimplicialMap(path, path, {0: 1, 1: 1, 2: 2, 3: 2})
    second = SimplicialMap(path, path, {0: 1, 1: 1, 2: 2, 3: 3})

    certificate = simplicial_homotopy_equivalence_certificate(
        forward, backward, source_homotopy=(first, second)
    )

    assert certificate is not None


def test_simplicial_homotopy_equivalence_rejects_invalid_composites():
    from pycombtop import simplicial_homotopy_equivalence_certificate

    path = SimplicialComplex({0, 1, 2}, [{0, 1}, {1, 2}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, dict.fromkeys(path.vertex_set, "*"))
    backward = SimplicialMap(point, path, {"*": 0})

    assert simplicial_homotopy_equivalence_certificate(forward, backward) is None


def test_simplicial_homotopy_equivalence_verifier_rejects_tampering():
    from dataclasses import replace

    from pycombtop import (
        simplicial_homotopy_equivalence_certificate,
        verify_simplicial_homotopy_equivalence,
    )

    path = SimplicialComplex({0, 1, 2}, [{0, 1}, {1, 2}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, dict.fromkeys(path.vertex_set, "*"))
    backward = SimplicialMap(point, path, {"*": 1})
    certificate = simplicial_homotopy_equivalence_certificate(forward, backward)

    assert certificate is not None
    assert not verify_simplicial_homotopy_equivalence(point, path, certificate)
    assert not verify_simplicial_homotopy_equivalence(path, point, object())
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, source_homotopy=())
    )


def test_simplicial_homotopy_equivalence_requires_opposite_map_endpoints():
    from pycombtop import simplicial_homotopy_equivalence_certificate

    path = SimplicialComplex({0, 1}, [{0, 1}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, {0: "*", 1: "*"})
    wrong_direction = SimplicialMap(path, point, {0: "*", 1: "*"})

    with pytest.raises(ValueError, match="opposite endpoints"):
        simplicial_homotopy_equivalence_certificate(forward, wrong_direction)


def test_simplicial_homotopy_equivalence_verifier_checks_chain_structure():
    from dataclasses import replace

    from pycombtop import (
        simplicial_homotopy_equivalence_certificate,
        verify_simplicial_homotopy_equivalence,
    )

    path = SimplicialComplex({0, 1, 2, 3}, [{0, 1}, {1, 2}, {2, 3}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, dict.fromkeys(path.vertex_set, "*"))
    backward = SimplicialMap(point, path, {"*": 1})
    first = SimplicialMap(path, path, {0: 1, 1: 1, 2: 2, 3: 2})
    second = SimplicialMap(path, path, {0: 1, 1: 1, 2: 2, 3: 3})
    certificate = simplicial_homotopy_equivalence_certificate(
        forward, backward, source_homotopy=(first, second)
    )

    assert certificate is not None
    start, end = certificate.source_homotopy[0], certificate.source_homotopy[-1]
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, source_homotopy=[start, end])
    )
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, source_homotopy=(start, object(), end))
    )

    wrong_domain = SimplicialMap(point, point, {"*": "*"})
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, source_homotopy=(start, wrong_domain, end))
    )


def test_simplicial_homotopy_equivalence_verifier_checks_contiguity_and_target_chain():
    from dataclasses import replace

    from pycombtop import (
        simplicial_homotopy_equivalence_certificate,
        verify_simplicial_homotopy_equivalence,
    )

    path = SimplicialComplex({0, 1, 2, 3}, [{0, 1}, {1, 2}, {2, 3}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, dict.fromkeys(path.vertex_set, "*"))
    backward = SimplicialMap(point, path, {"*": 1})
    first = SimplicialMap(path, path, {0: 1, 1: 1, 2: 2, 3: 2})
    second = SimplicialMap(path, path, {0: 1, 1: 1, 2: 2, 3: 3})
    certificate = simplicial_homotopy_equivalence_certificate(
        forward, backward, source_homotopy=(first, second)
    )

    assert certificate is not None
    start, end = certificate.source_homotopy[0], certificate.source_homotopy[-1]
    noncontiguous = SimplicialMap(
        path, path, {vertex: vertex for vertex in path.vertex_set}
    )
    assert not are_contiguous(start, noncontiguous)
    assert not verify_simplicial_homotopy_equivalence(
        path,
        point,
        replace(certificate, source_homotopy=(start, noncontiguous, end)),
    )
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, target_homotopy=())
    )


def test_simplicial_homotopy_equivalence_rejects_malformed_maps_and_spaces():
    from dataclasses import replace
    from typing import Any, cast

    from pycombtop import (
        simplicial_homotopy_equivalence_certificate,
        verify_simplicial_homotopy_equivalence,
    )

    path = SimplicialComplex({0, 1}, [{0, 1}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, {0: "*", 1: "*"})
    backward = SimplicialMap(point, path, {"*": 0})
    certificate = simplicial_homotopy_equivalence_certificate(forward, backward)

    with pytest.raises(TypeError, match="forward and backward"):
        simplicial_homotopy_equivalence_certificate(cast(Any, object()), backward)
    with pytest.raises(TypeError, match="forward and backward"):
        simplicial_homotopy_equivalence_certificate(forward, cast(Any, object()))

    assert certificate is not None
    assert not verify_simplicial_homotopy_equivalence(
        cast(Any, object()), point, certificate
    )
    assert not verify_simplicial_homotopy_equivalence(
        path, cast(Any, object()), certificate
    )
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, forward=cast(Any, object()))
    )
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, backward=cast(Any, object()))
    )

def test_simplicial_map_composition_rejects_incompatible_endpoints():
    from pycombtop.simplicial_maps import _compose_simplicial_maps

    first_space = SimplicialComplex({0}, [{0}])
    middle_space = SimplicialComplex({1}, [{1}])
    last_space = SimplicialComplex({"*"}, [{"*"}])
    first = SimplicialMap(first_space, first_space, {0: 0})
    second = SimplicialMap(middle_space, last_space, {1: "*"})

    with pytest.raises(ValueError, match="target of the first map"):
        _compose_simplicial_maps(first, second)


def test_simplicial_homotopy_equivalence_verifier_checks_both_chain_endpoints():
    from dataclasses import replace

    from pycombtop import (
        simplicial_homotopy_equivalence_certificate,
        verify_simplicial_homotopy_equivalence,
    )

    path = SimplicialComplex({0, 1}, [{0, 1}])
    point = SimplicialComplex({"*"}, [{"*"}])
    forward = SimplicialMap(path, point, {0: "*", 1: "*"})
    backward = SimplicialMap(point, path, {"*": 0})
    certificate = simplicial_homotopy_equivalence_certificate(forward, backward)

    assert certificate is not None
    source_chain = certificate.source_homotopy
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, source_homotopy=(source_chain[-1],))
    )
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, source_homotopy=(source_chain[0],))
    )



def test_simplicial_homotopy_equivalence_rejects_noncontiguous_source_and_target_chains():
    from pycombtop import simplicial_homotopy_equivalence_certificate

    path = SimplicialComplex({0, 1, 2}, [{0, 1}, {1, 2}])
    point = SimplicialComplex({"*"}, [{"*"}])
    collapse = SimplicialMap(path, point, dict.fromkeys(path.vertex_set, "*"))
    inclusion = SimplicialMap(point, path, {"*": 1})

    constant_at_zero = SimplicialMap(
        path, path, dict.fromkeys(path.vertex_set, 0)
    )
    assert simplicial_homotopy_equivalence_certificate(
        collapse, inclusion, source_homotopy=(constant_at_zero,)
    ) is None

    assert simplicial_homotopy_equivalence_certificate(
        inclusion, collapse, target_homotopy=(constant_at_zero,)
    ) is None


def test_simplicial_homotopy_equivalence_verifier_rejects_wrong_chain_codomain():
    from dataclasses import replace

    from pycombtop import (
        simplicial_homotopy_equivalence_certificate,
        verify_simplicial_homotopy_equivalence,
    )

    path = SimplicialComplex({0, 1}, [{0, 1}])
    point = SimplicialComplex({"*"}, [{"*"}])
    other_point = SimplicialComplex({"x"}, [{"x"}])
    forward = SimplicialMap(path, point, {0: "*", 1: "*"})
    backward = SimplicialMap(point, path, {"*": 0})
    certificate = simplicial_homotopy_equivalence_certificate(forward, backward)
    assert certificate is not None

    wrong_codomain = SimplicialMap(path, other_point, {0: "x", 1: "x"})
    chain = (certificate.source_homotopy[0], wrong_codomain, certificate.source_homotopy[-1])
    assert not verify_simplicial_homotopy_equivalence(
        path, point, replace(certificate, source_homotopy=chain)
    )
