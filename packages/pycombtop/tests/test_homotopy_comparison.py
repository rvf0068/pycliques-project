"""Tests for conservative comparisons of simplicial homotopy types."""

import pycombtop
import pytest


def circle() -> pycombtop.SimplicialComplex:
    return pycombtop.SimplicialComplex(
        {0, 1, 2},
        [{0, 1}, {1, 2}, {0, 2}],
    )


def point() -> pycombtop.SimplicialComplex:
    return pycombtop.SimplicialComplex({"p"}, [{"p"}])


def test_homology_mismatch_proves_not_homotopy_equivalent():
    result = pycombtop.compare_homotopy(circle(), point(), backend="python")

    assert result.status is pycombtop.HomotopyComparisonStatus.PROVED_NOT_EQUIVALENT
    assert result.is_equivalent is False
    assert result.field_profiles[0] == (2, (0, 1), ())
    assert "GF(2)" in result.reason


def test_matching_homology_without_certificate_is_inconclusive():
    source = circle()
    target = pycombtop.SimplicialComplex(
        {"a", "b", "c", "d"},
        [{"a", "b"}, {"b", "c"}, {"c", "d"}, {"d", "a"}],
    )

    result = pycombtop.compare_homotopy(source, target, backend="python")

    assert result.status is pycombtop.HomotopyComparisonStatus.INCONCLUSIVE
    assert result.is_equivalent is None
    assert result.field_profiles == ((2, (0, 1), (0, 1)), ("Q", (0, 1), (0, 1)))
    assert "does not prove" in result.reason


def test_verified_certificate_proves_equivalence_without_homology():
    source = pycombtop.SimplicialComplex({0, 1}, [{0, 1}])
    target = point()
    forward = pycombtop.SimplicialMap(source, target, {0: "p", 1: "p"})
    backward = pycombtop.SimplicialMap(target, source, {"p": 0})
    certificate = pycombtop.simplicial_homotopy_equivalence_certificate(
        forward, backward
    )

    result = pycombtop.compare_homotopy(
        source,
        target,
        certificate=certificate,
        max_faces=1,
        backend="python",
    )

    assert result.status is pycombtop.HomotopyComparisonStatus.PROVED_EQUIVALENT
    assert result.is_equivalent is True
    assert result.certificate is certificate
    assert result.field_profiles == ()


def test_invalid_certificate_falls_back_to_homology_obstruction():
    edge = pycombtop.SimplicialComplex({0, 1}, [{0, 1}])
    target = point()
    forward = pycombtop.SimplicialMap(edge, target, {0: "p", 1: "p"})
    backward = pycombtop.SimplicialMap(target, edge, {"p": 0})
    certificate = pycombtop.simplicial_homotopy_equivalence_certificate(
        forward, backward
    )

    result = pycombtop.compare_homotopy(
        circle(),
        target,
        certificate=certificate,
        backend="python",
    )

    assert result.status is pycombtop.HomotopyComparisonStatus.PROVED_NOT_EQUIVALENT
    assert result.diagnostic == (
        "The supplied homotopy-equivalence certificate did not verify."
    )


def test_homology_resource_limit_is_inconclusive():
    result = pycombtop.compare_homotopy(
        circle(), point(), max_faces=5, backend="python"
    )

    assert result.status is pycombtop.HomotopyComparisonStatus.INCONCLUSIVE
    assert result.is_equivalent is None
    assert "resource limit" in result.reason


def test_fields_must_be_nonempty_and_unique():
    source = circle()

    with pytest.raises(ValueError, match="at least one"):
        pycombtop.compare_homotopy(source, source, fields=())

    with pytest.raises(ValueError, match="duplicates"):
        pycombtop.compare_homotopy(source, source, fields=(2, 2))


def test_empty_realizations_are_rejected():
    empty_complexes = (
        pycombtop.SimplicialComplex(set(), []),
        pycombtop.SimplicialComplex(set(), [set()]),
    )

    for empty in empty_complexes:
        with pytest.raises(ValueError, match="nonempty realizations"):
            pycombtop.compare_homotopy(empty, point())


def test_fields_are_validated_even_when_certificate_verifies():
    source = pycombtop.SimplicialComplex({0, 1}, [{0, 1}])
    target = point()
    forward = pycombtop.SimplicialMap(source, target, {0: "p", 1: "p"})
    backward = pycombtop.SimplicialMap(target, source, {"p": 0})
    certificate = pycombtop.simplicial_homotopy_equivalence_certificate(
        forward, backward
    )

    with pytest.raises(ValueError, match="at least one"):
        pycombtop.compare_homotopy(source, target, certificate=certificate, fields=())

    with pytest.raises(ValueError, match="duplicates"):
        pycombtop.compare_homotopy(
            source, target, certificate=certificate, fields=(2, 2)
        )

    with pytest.raises(ValueError, match="prime integer"):
        pycombtop.compare_homotopy(source, target, certificate=certificate, fields=(4,))


def test_resource_limit_preserves_rejected_certificate_diagnostic():
    source = circle()
    target = point()
    edge = pycombtop.SimplicialComplex({0, 1}, [{0, 1}])
    forward = pycombtop.SimplicialMap(edge, target, {0: "p", 1: "p"})
    backward = pycombtop.SimplicialMap(target, edge, {"p": 0})
    certificate = pycombtop.simplicial_homotopy_equivalence_certificate(
        forward, backward
    )

    result = pycombtop.compare_homotopy(
        source,
        target,
        certificate=certificate,
        max_faces=5,
        backend="python",
    )

    assert result.status is pycombtop.HomotopyComparisonStatus.INCONCLUSIVE
    assert result.diagnostic is not None
    assert "certificate did not verify" in result.diagnostic
    assert "face budget" in result.diagnostic
