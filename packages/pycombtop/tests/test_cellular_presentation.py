"""Homotopy-preserving cellular presentation regression tests."""

from dataclasses import replace

import pytest

from pycombtop import (
    CellularPresentation,
    PresentationElimination,
    SimplicialComplex,
    cellular_presentation,
    reduce_cellular_presentation,
    verify_presentation_reduction,
)


def test_elimination_preserves_duplicate_and_empty_two_cells():
    source = CellularPresentation((1, 2), ((1, 2), (2, 2), (), (2, 2)))
    result = reduce_cellular_presentation(source)
    assert result.reduced == CellularPresentation((2,), ((2, 2), (), (2, 2)))
    assert verify_presentation_reduction(source, result)
    assert not verify_presentation_reduction(
        source, replace(result, reduced=CellularPresentation((2,), ((2, 2),)))
    )


def test_signed_noncommutative_substitution():
    source = CellularPresentation((1, 2, 3), ((2, -1, 3), (1, 1), (-1, -1)))
    result = reduce_cellular_presentation(source)
    assert result.steps == (PresentationElimination(1, 0),)
    assert result.reduced == CellularPresentation(
        (2, 3), ((3, 2, 3, 2), (-2, -3, -2, -3))
    )
    assert verify_presentation_reduction(source, result)
    assert not verify_presentation_reduction(
        source, replace(result, steps=(PresentationElimination(1, 1),))
    )


def test_projective_plane_triangulation():
    facets = [
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 5),
        (1, 4, 6),
        (1, 5, 6),
        (2, 3, 6),
        (2, 4, 5),
        (2, 5, 6),
        (3, 4, 5),
        (3, 4, 6),
    ]
    source = cellular_presentation(SimplicialComplex(range(1, 7), facet_set=facets))
    assert len(source.generators) == len(source.relators) == 10
    result = reduce_cellular_presentation(source)
    assert verify_presentation_reduction(source, result)
    assert len(result.reduced.generators) == len(result.reduced.relators) == 1
    (word,) = result.reduced.relators
    assert len(word) == 2 and word[0] == word[1]


def test_torus_commutator_is_not_cancelled_as_an_abelian_relation():
    source = CellularPresentation((1, 2), ((1, 2, -1, -2),))
    result = reduce_cellular_presentation(source)
    assert result.reduced == source
    assert not result.steps


def test_dimension_and_connectivity_preconditions():
    with pytest.raises(ValueError):
        cellular_presentation(SimplicialComplex(range(4), facet_set=[range(4)]))
    with pytest.raises(ValueError):
        cellular_presentation(SimplicialComplex(range(2), facet_set=[(0,), (1,)]))


def test_budget_returns_verifiable_partial_reduction():
    source = CellularPresentation((1, 2), ((1, 2), (2, 2)))
    result = reduce_cellular_presentation(source, max_total_letters=1)
    assert result.limited
    assert verify_presentation_reduction(source, result)
