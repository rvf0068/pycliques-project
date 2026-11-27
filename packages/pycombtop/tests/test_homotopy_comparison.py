"""Tests for conservative comparisons of simplicial homotopy types."""

import pytest

from pycombtop import (
    compare_homotopy,
    HomotopyComparisonStatus,
    simplicial_homotopy_equivalence_certificate,
    SimplicialComplex,
    SimplicialMap,
)


def circle() -> SimplicialComplex:
    return SimplicialComplex(
        {0, 1, 2},
        [{0, 1}, {1, 2}, {0, 2}],
    )


def point() -> SimplicialComplex:
    return SimplicialComplex({"p"}, [{"p"}])


def test_homology_mismatch_proves_not_homotopy_equivalent():
    result = compare_homotopy(circle(), point(), backend="python")

    assert result.status is HomotopyComparisonStatus.PROVED_NOT_EQUIVALENT
    assert result.is_equivalent is False
    assert result.field_profiles[0] == (2, (0, 1), ())
    assert "GF(2)" in result.reason


def test_matching_homology_without_certificate_is_inconclusive():
    source = circle()
    target = SimplicialComplex(
        {"a", "b", "c", "d"},
        [{"a", "b"}, {"b", "c"}, {"c", "d"}, {"d", "a"}],
    )

    result = compare_homotopy(source, target, backend="python")

    assert result.status is HomotopyComparisonStatus.INCONCLUSIVE
    assert result.is_equivalent is None
    assert result.field_profiles == ((2, (0, 1), (0, 1)), ("Q", (0, 1), (0, 1)))
    assert "does not prove" in result.reason


def test_verified_certificate_proves_equivalence_without_homology():
    source = SimplicialComplex({0, 1}, [{0, 1}])
    target = point()
    forward = SimplicialMap(source, target, {0: "p", 1: "p"})
    backward = SimplicialMap(target, source, {"p": 0})
    certificate = simplicial_homotopy_equivalence_certificate(forward, backward)

    result = compare_homotopy(
        source,
        target,
        certificate=certificate,
        max_faces=0,
        backend="python",
    )

    assert result.status is HomotopyComparisonStatus.PROVED_EQUIVALENT
    assert result.is_equivalent is True
    assert result.certificate is certificate
    assert result.field_profiles == ()


def test_invalid_certificate_falls_back_to_homology_obstruction():
    edge = SimplicialComplex({0, 1}, [{0, 1}])
    target = point()
    forward = SimplicialMap(edge, target, {0: "p", 1: "p"})
    backward = SimplicialMap(target, edge, {"p": 0})
    certificate = simplicial_homotopy_equivalence_certificate(forward, backward)

    result = compare_homotopy(
        circle(),
        target,
        certificate=certificate,
        backend="python",
    )

    assert result.status is HomotopyComparisonStatus.PROVED_NOT_EQUIVALENT
    assert result.diagnostic == (
        "The supplied homotopy-equivalence certificate did not verify."
    )


def test_homology_resource_limit_is_inconclusive():
    result = compare_homotopy(circle(), point(), max_faces=0, backend="python")

    assert result.status is HomotopyComparisonStatus.INCONCLUSIVE
    assert result.is_equivalent is None
    assert "resource limit" in result.reason


def test_fields_must_be_nonempty_and_unique():
    source = circle()

    with pytest.raises(ValueError, match="at least one"):
        compare_homotopy(source, source, fields=())

    with pytest.raises(ValueError, match="duplicates"):
        compare_homotopy(source, source, fields=(2, 2))


def test_empty_realizations_are_rejected():
    empty = SimplicialComplex(set(), [])

    with pytest.raises(ValueError, match="nonempty realizations"):
        compare_homotopy(empty, point())
