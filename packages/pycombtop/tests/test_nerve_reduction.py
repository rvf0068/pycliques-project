"""Tests of alternating nerves using explicit replay."""

from pycombtop.nerve_reduction import (
    NerveReduction,
    alternating_nerve_reduction,
    verify_nerve_reduction,
)


def test_reduction_to_simplex():
    # Four triangles with a shared vertex; their nerve is a simplex.
    facets = [7, 13, 25, 49]
    result = alternating_nerve_reduction(facets)
    assert result.proved_contractible
    assert result.facet_counts == (4, 1)
    assert verify_nerve_reduction(facets, result)
    assert not verify_nerve_reduction(facets, NerveReduction((1,), (4, 1)))


def test_cycle_and_empty():
    for facets in ([3, 6, 5], [0], []):
        result = alternating_nerve_reduction(facets)
        assert not result.proved_contractible
        assert verify_nerve_reduction(facets, result)


def test_redundant_facets():
    result = alternating_nerve_reduction([0, 1, 3, 3])
    assert result == NerveReduction((3,), (1,))
    assert verify_nerve_reduction([0, 1, 3, 3], result)
