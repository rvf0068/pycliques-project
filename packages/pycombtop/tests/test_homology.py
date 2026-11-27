"""Tests for the unified exact simplicial homology interface."""

from __future__ import annotations

import pytest

from pycombtop.homology import simplicial_homology
from pycombtop.homology_backend import HomologyResourceLimit
from pycombtop.simplex import SimplicialComplex


RP2_FACETS = [
    {1, 2, 3},
    {1, 2, 4},
    {1, 3, 5},
    {1, 4, 6},
    {1, 5, 6},
    {2, 3, 6},
    {2, 4, 5},
    {2, 5, 6},
    {3, 4, 5},
    {3, 4, 6},
]


def test_portable_backend_handles_basic_complexes_and_reduced_conventions():
    triangle = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    circle = SimplicialComplex(
        {0, 1, 2},
        [{0, 1}, {1, 2}, {0, 2}],
    )
    disconnected = SimplicialComplex({"left", "right"}, [{"left"}, {"right"}])

    triangle_result = simplicial_homology(triangle, backend="python")
    assert triangle_result.betti_numbers == (0, 0, 0)
    assert simplicial_homology(circle, backend="python").betti_numbers == (0, 1)
    assert simplicial_homology(disconnected, backend="python").betti_numbers == (1,)
    assert simplicial_homology(
        disconnected, backend="python", reduced=False
    ).betti_numbers == (2,)


def test_exact_coefficients_distinguish_projective_plane_torsion():
    projective_plane = SimplicialComplex({1, 2, 3, 4, 5, 6}, RP2_FACETS)

    mod_two = simplicial_homology(
        projective_plane, field=2, backend="python"
    )
    mod_three = simplicial_homology(
        projective_plane, field=3, backend="python"
    )
    rational = simplicial_homology(
        projective_plane, field="Q", backend="python"
    )

    assert mod_two.betti_numbers == (0, 1, 1)
    assert mod_three.betti_numbers == (0, 0, 0)
    assert rational.betti_numbers == (0, 0, 0)
    assert rational.field == "Q"
    assert rational.backend == "python"


def test_relative_homology_of_a_pair():
    disk = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    boundary = SimplicialComplex(
        {0, 1, 2},
        [{0, 1}, {1, 2}, {0, 2}],
    )
    point = SimplicialComplex({0}, [{0}])

    disk_pair = simplicial_homology(
        disk, relative_to=boundary, field="Q", backend="python"
    )
    circle_pair = simplicial_homology(
        boundary, relative_to=point, field="Q", backend="python"
    )
    reduced_empty_pair = simplicial_homology(
        boundary,
        relative_to=SimplicialComplex(set(), []),
        field="Q",
        backend="python",
    )

    assert disk_pair.betti_numbers == (0, 0, 1)
    assert disk_pair.relative is True
    assert circle_pair.betti_numbers == (0, 1)
    assert reduced_empty_pair.betti_numbers == (0, 1)


def test_gudhi_and_python_backends_agree_when_gudhi_is_available():
    pytest.importorskip("gudhi")
    projective_plane = SimplicialComplex({1, 2, 3, 4, 5, 6}, RP2_FACETS)
    for field in (2, 3, 5):
        accelerated = simplicial_homology(
            projective_plane, field=field, backend="gudhi"
        )
        portable = simplicial_homology(
            projective_plane, field=field, backend="python"
        )
        assert accelerated.betti_numbers == portable.betti_numbers


def test_auto_backend_falls_back_when_gudhi_is_unavailable(monkeypatch):
    complex_ = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])

    def unavailable(*_args, **_kwargs):
        raise ImportError("synthetic missing GUDHI")

    monkeypatch.setitem(
        simplicial_homology.__globals__, "gudhi_betti_facets", unavailable
    )
    result = simplicial_homology(complex_, backend="auto")
    assert result.backend == "python"
    assert result.betti_numbers == (0, 0, 0)
    with pytest.raises(ImportError, match="synthetic"):
        simplicial_homology(complex_, backend="gudhi")


def test_face_budget_overflow_is_never_reported_as_zero_homology():
    triangle = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    with pytest.raises(HomologyResourceLimit):
        simplicial_homology(triangle, backend="python", max_faces=6)
    assert simplicial_homology(
        triangle, backend="python", max_faces=7
    ).betti_numbers == (0, 0, 0)


def test_empty_complex_conventions_and_ghost_labels():
    void = SimplicialComplex({"ghost"}, [])
    empty_face = SimplicialComplex(set(), [set()])
    ghosted = SimplicialComplex({"actual", "ghost"}, [{"actual"}])

    assert simplicial_homology(void, backend="python").betti_numbers == ()
    assert simplicial_homology(empty_face, backend="python").betti_numbers == ()
    assert simplicial_homology(ghosted, backend="python").betti_numbers == (0,)


def test_invalid_inputs_and_non_subcomplex_are_rejected():
    triangle = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    unrelated = SimplicialComplex({0, 1, 3}, [{0, 1, 3}])

    with pytest.raises(TypeError, match="complex_"):
        simplicial_homology([{0, 1}], backend="python")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="subcomplex"):
        simplicial_homology(triangle, relative_to=unrelated, backend="python")
    with pytest.raises(ValueError, match="prime"):
        simplicial_homology(triangle, field=4, backend="python")
    with pytest.raises(ValueError, match="prime"):
        simplicial_homology(triangle, field=True, backend="python")
    with pytest.raises(ValueError, match="max_faces"):
        simplicial_homology(triangle, max_faces=0, backend="python")
    with pytest.raises(ValueError, match="backend"):
        simplicial_homology(triangle, backend="numpy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="degree"):
        simplicial_homology(triangle, backend="python").betti_number(-1)


def test_ambiguous_vertex_labels_require_an_explicit_order():
    class SameRepr:
        def __repr__(self):
            return "same"

    first, second = SameRepr(), SameRepr()
    complex_ = SimplicialComplex({first, second}, [{first, second}])

    with pytest.raises(ValueError, match="ambiguous stable keys"):
        simplicial_homology(complex_, backend="python")
    result = simplicial_homology(
        complex_, backend="python", vertex_order=[first, second]
    )
    assert result.betti_numbers == (0, 0)
