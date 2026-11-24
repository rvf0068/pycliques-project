"""Tests for the facet-indexed relative collapse engine."""

import pytest

from pycombtop.indexed_relative_collapse import (
    indexed_relative_collapse,
    verify_indexed_relative_collapse,
)
from pycombtop.relative_collapse import CollapseStatus, verify_relative_collapse
from pycombtop.simplex import Simplex, SimplicialComplex


def test_relative_collapse_and_independent_replay():
    source = SimplicialComplex(range(4), facet_set=[{0, 1, 2}, {1, 2, 3}])
    target = SimplicialComplex(range(4), facet_set=[{0, 1}, {1, 2}, {2, 3}])
    result = indexed_relative_collapse(source, target)
    assert result.status is CollapseStatus.PROVED
    assert len(result.pairs) == 2
    assert verify_indexed_relative_collapse(source, target, result.pairs)
    assert verify_relative_collapse(source, target, result.pairs)
    assert not verify_indexed_relative_collapse(source, target, result.pairs[:-1])
    assert not verify_indexed_relative_collapse(
        source, target, [(Simplex({0}), Simplex({0, 1}))]
    )
    assert not verify_indexed_relative_collapse(
        source, target, [(Simplex({9}), Simplex({0, 9}))]
    )
    assert not verify_indexed_relative_collapse(source, target, [(Simplex({0}),)])


def test_large_preserved_simplex_is_not_expanded():
    target = SimplicialComplex(range(100), facet_set=[set(range(100))])
    source = SimplicialComplex(
        range(102), facet_set=[set(range(100)), {0, 100, 101}]
    )
    result = indexed_relative_collapse(source, target, max_bad_faces=6)
    assert result.status is CollapseStatus.PROVED
    assert len(result.pairs) == 3
    limited = indexed_relative_collapse(source, target, max_bad_faces=5)
    assert limited.status is CollapseStatus.INCONCLUSIVE
    assert not verify_indexed_relative_collapse(
        source, target, result.pairs, max_bad_faces=5
    )
    identical = indexed_relative_collapse(target, target, max_bad_faces=0)
    assert identical.status is CollapseStatus.PROVED


def test_dead_end_and_bad_target():
    source = SimplicialComplex(range(3), facet_set=[{0, 1}, {1, 2}, {0, 2}])
    target = SimplicialComplex(range(3), facet_set=[{0}])
    result = indexed_relative_collapse(source, target)
    assert result.status is CollapseStatus.INCONCLUSIVE
    with pytest.raises(ValueError):
        indexed_relative_collapse(
            source, SimplicialComplex(range(3), facet_set=[{0, 1, 2}])
        )
    with pytest.raises(ValueError):
        indexed_relative_collapse(
            source, SimplicialComplex(range(4), facet_set=[{3}])
        )
    with pytest.raises(ValueError):
        indexed_relative_collapse(source, target, max_bad_faces=-1)
    with pytest.raises(ValueError):
        verify_indexed_relative_collapse(source, target, [], max_bad_faces=-1)
    void = SimplicialComplex([], facet_set=[])
    assert indexed_relative_collapse(target, void).status is CollapseStatus.INCONCLUSIVE
    assert not verify_indexed_relative_collapse(target, void, [])
