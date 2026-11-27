"""Tests for the unified exact simplicial homology interface."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

pycombtop: Any = import_module("pycombtop")


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
    triangle = pycombtop.SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    circle = pycombtop.SimplicialComplex(
        {0, 1, 2},
        [{0, 1}, {1, 2}, {0, 2}],
    )
    disconnected = pycombtop.SimplicialComplex({"left", "right"}, [{"left"}, {"right"}])

    triangle_result = pycombtop.simplicial_homology(triangle, backend="python")
    assert triangle_result.betti_numbers == (0, 0, 0)
    circle_result = pycombtop.simplicial_homology(circle, backend="python")
    disconnected_result = pycombtop.simplicial_homology(disconnected, backend="python")
    assert circle_result.betti_numbers == (0, 1)
    assert disconnected_result.betti_numbers == (1,)
    assert pycombtop.simplicial_homology(
        disconnected, backend="python", reduced=False
    ).betti_numbers == (2,)


def test_exact_coefficients_distinguish_projective_plane_torsion():
    projective_plane = pycombtop.SimplicialComplex({1, 2, 3, 4, 5, 6}, RP2_FACETS)

    mod_two = pycombtop.simplicial_homology(projective_plane, field=2, backend="python")
    mod_three = pycombtop.simplicial_homology(
        projective_plane, field=3, backend="python"
    )
    rational = pycombtop.simplicial_homology(
        projective_plane, field="Q", backend="python"
    )

    assert mod_two.betti_numbers == (0, 1, 1)
    assert mod_three.betti_numbers == (0, 0, 0)
    assert rational.betti_numbers == (0, 0, 0)
    assert rational.field == "Q"
    assert rational.backend == "python"


def test_relative_homology_of_a_pair():
    disk = pycombtop.SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    boundary = pycombtop.SimplicialComplex(
        {0, 1, 2},
        [{0, 1}, {1, 2}, {0, 2}],
    )
    point = pycombtop.SimplicialComplex({0}, [{0}])

    disk_pair = pycombtop.simplicial_homology(
        disk, relative_to=boundary, field="Q", backend="python"
    )
    circle_pair = pycombtop.simplicial_homology(
        boundary, relative_to=point, field="Q", backend="python"
    )
    reduced_empty_pair = pycombtop.simplicial_homology(
        boundary,
        relative_to=pycombtop.SimplicialComplex(set(), []),
        field="Q",
        backend="python",
    )

    assert disk_pair.betti_numbers == (0, 0, 1)
    assert disk_pair.relative is True
    assert circle_pair.betti_numbers == (0, 1)
    assert reduced_empty_pair.betti_numbers == (0, 1)


def test_gudhi_and_python_backends_agree_when_gudhi_is_available():
    pytest.importorskip("gudhi")
    projective_plane = pycombtop.SimplicialComplex({1, 2, 3, 4, 5, 6}, RP2_FACETS)
    for field in (2, 3, 5):
        accelerated = pycombtop.simplicial_homology(
            projective_plane, field=field, backend="gudhi"
        )
        portable = pycombtop.simplicial_homology(
            projective_plane, field=field, backend="python"
        )
        assert accelerated.betti_numbers == portable.betti_numbers


def test_auto_backend_falls_back_when_gudhi_is_unavailable(monkeypatch):
    complex_ = pycombtop.SimplicialComplex({0, 1, 2}, [{0, 1, 2}])

    def unavailable(*_args, **_kwargs):
        raise ImportError("synthetic missing GUDHI")

    monkeypatch.setitem(
        pycombtop.simplicial_homology.__globals__,
        "gudhi_betti_facets",
        unavailable,
    )
    result = pycombtop.simplicial_homology(complex_, backend="auto")
    assert result.backend == "python"
    assert result.betti_numbers == (0, 0, 0)
    with pytest.raises(ImportError, match="synthetic"):
        pycombtop.simplicial_homology(complex_, backend="gudhi")


def test_face_budget_overflow_is_never_reported_as_zero_homology():
    triangle = pycombtop.SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    with pytest.raises(pycombtop.HomologyResourceLimit):
        pycombtop.simplicial_homology(triangle, backend="python", max_faces=6)
    assert pycombtop.simplicial_homology(
        triangle, backend="python", max_faces=7
    ).betti_numbers == (0, 0, 0)


def test_empty_complex_conventions_and_ghost_labels():
    void = pycombtop.SimplicialComplex({"ghost"}, [])
    empty_face = pycombtop.SimplicialComplex(set(), [set()])
    ghosted = pycombtop.SimplicialComplex({"actual", "ghost"}, [{"actual"}])

    assert pycombtop.simplicial_homology(void, backend="python").betti_numbers == ()
    empty_face_result = pycombtop.simplicial_homology(empty_face, backend="python")
    ghosted_result = pycombtop.simplicial_homology(ghosted, backend="python")
    assert empty_face_result.betti_numbers == ()
    assert ghosted_result.betti_numbers == (0,)


def test_invalid_inputs_and_non_subcomplex_are_rejected():
    triangle = pycombtop.SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
    unrelated = pycombtop.SimplicialComplex({0, 1, 3}, [{0, 1, 3}])

    with pytest.raises(TypeError, match="complex_"):
        pycombtop.simplicial_homology(  # type: ignore[arg-type]
            [{0, 1}], backend="python"
        )
    with pytest.raises(ValueError, match="subcomplex"):
        pycombtop.simplicial_homology(triangle, relative_to=unrelated, backend="python")
    with pytest.raises(ValueError, match="prime"):
        pycombtop.simplicial_homology(triangle, field=4, backend="python")
    with pytest.raises(ValueError, match="prime"):
        pycombtop.simplicial_homology(triangle, field=True, backend="python")
    with pytest.raises(ValueError, match="max_faces"):
        pycombtop.simplicial_homology(triangle, max_faces=0, backend="python")
    with pytest.raises(ValueError, match="backend"):
        pycombtop.simplicial_homology(  # type: ignore[arg-type]
            triangle, backend="numpy"
        )
    with pytest.raises(ValueError, match="degree"):
        pycombtop.simplicial_homology(triangle, backend="python").betti_number(-1)


def test_ambiguous_vertex_labels_require_an_explicit_order():
    class SameRepr:
        def __repr__(self):
            return "same"

    first, second = SameRepr(), SameRepr()
    complex_ = pycombtop.SimplicialComplex({first, second}, [{first, second}])

    with pytest.raises(ValueError, match="ambiguous stable keys"):
        pycombtop.simplicial_homology(complex_, backend="python")
    result = pycombtop.simplicial_homology(
        complex_, backend="python", vertex_order=[first, second]
    )
    assert result.betti_numbers == (0, 0)
