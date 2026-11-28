import pytest
from pycombtop import PermutationAction, SimplicialComplex
from sympy.combinatorics import Permutation

from pyhomrep import EquivariantHomology


def tetrahedron_boundary():
    vertices = ("a", "b", "c", "d")
    facets = [set(vertices) - {vertex} for vertex in vertices]
    return SimplicialComplex(vertices, facet_set=facets)


def test_exact_character_on_tetrahedron_boundary():
    complex_ = tetrahedron_boundary()
    transposition = Permutation([1, 0, 2, 3])
    three_cycle = Permutation([1, 2, 0, 3])
    four_cycle = Permutation([1, 2, 3, 0])
    action = PermutationAction(("a", "b", "c", "d"), [transposition, four_cycle])
    homology = EquivariantHomology(complex_, action)

    assert action.group.order() == 24
    assert homology.character(Permutation(size=4), 0) == 1
    assert homology.character(transposition, 0) == 1
    assert homology.character(transposition, 2) == -1
    assert homology.character(three_cycle, 2) == 1
    assert homology.character(four_cycle, 2) == -1
    assert homology.character(transposition, 1) == 0


def test_exact_reduced_character_on_disconnected_complex():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    swap = Permutation([1, 0])
    action = PermutationAction((0, 1), [swap])
    homology = EquivariantHomology(complex_, action, reduced=True)

    assert homology.character(Permutation(size=2), 0) == 1
    assert homology.character(swap, 0) == -1
    assert homology.character(swap, 1) == 0


def test_character_requires_group_element():
    complex_ = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}, {1, 2}, {0, 2}])
    rotation = Permutation([1, 2, 0])
    action = PermutationAction((0, 1, 2), [rotation])
    homology = EquivariantHomology(complex_, action)

    with pytest.raises(ValueError, match="not in the acting group"):
        homology.character(Permutation([1, 0, 2]), 1)
    with pytest.raises(ValueError, match="degree must be"):
        homology.character(rotation, -1)


def test_character_rejects_noninvariant_complex():
    complex_ = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}, {1, 2}])
    rotation = Permutation([1, 2, 0])
    action = PermutationAction((0, 1, 2), [rotation])

    with pytest.raises(ValueError, match="preserve the simplicial complex"):
        EquivariantHomology(complex_, action)
