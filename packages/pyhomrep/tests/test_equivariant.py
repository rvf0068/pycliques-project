import pytest
from pycombtop import PermutationAction, SimplicialComplex, SimplicialMap
from pyhomrep import EquivariantHomology
from sympy import Matrix
from sympy.combinatorics import Permutation


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


def test_exact_representation_matrices_on_tetrahedron_boundary():
    complex_ = tetrahedron_boundary()
    transposition = Permutation([1, 0, 2, 3])
    three_cycle = Permutation([1, 2, 0, 3])
    four_cycle = Permutation([1, 2, 3, 0])
    action = PermutationAction(("a", "b", "c", "d"), [transposition, four_cycle])
    homology = EquivariantHomology(complex_, action)

    assert homology.representation_matrix(Permutation(size=4), 0) == Matrix([[1]])
    assert homology.representation_matrix(transposition, 2) == Matrix([[-1]])
    assert homology.representation_matrix(three_cycle, 2) == Matrix([[1]])
    assert homology.representation_matrix(four_cycle, 2) == Matrix([[-1]])
    assert homology.representation_matrix(transposition, 1) == Matrix(0, 0, [])


def test_representation_matrix_on_contractible_interval():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    swap = Permutation([1, 0])
    action = PermutationAction((0, 1), [swap])
    homology = EquivariantHomology(complex_, action)

    assert homology.representation_matrix(Permutation(size=2), 0) == Matrix([[1]])
    assert homology.representation_matrix(swap, 0) == Matrix([[1]])
    assert homology.representation_matrix(swap, 1) == Matrix(0, 0, [])
    assert homology.representation_matrix(swap, 2) == Matrix(0, 0, [])


def test_representation_matrix_validates_element_and_degree():
    complex_ = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    rotation = Permutation([1, 2, 0])
    action = PermutationAction((0, 1, 2), [rotation])
    homology = EquivariantHomology(complex_, action)

    with pytest.raises(ValueError, match="degree must be"):
        homology.representation_matrix(rotation, -1)
    with pytest.raises(ValueError, match="not in the acting group"):
        homology.representation_matrix(Permutation([1, 0, 2]), 1)


def test_exact_reduced_homology_representation_matrix():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    swap = Permutation([1, 0])
    action = PermutationAction((0, 1), [swap])
    homology = EquivariantHomology(complex_, action, reduced=True)

    assert homology.representation_matrix(Permutation(size=2), 0) == Matrix([[1]])
    assert homology.representation_matrix(swap, 0) == Matrix([[-1]])


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


def test_equivariant_simplicial_self_map_induces_intertwining_homology_map():
    complex_ = SimplicialComplex(
        {0, 1, 2, 3},
        facet_set=[{0, 1}, {1, 2}, {2, 3}, {0, 3}],
    )
    reflection = Permutation([0, 3, 2, 1])
    action = PermutationAction((0, 1, 2, 3), [reflection])
    mapping = SimplicialMap(complex_, complex_, {0: 0, 1: 3, 2: 2, 3: 1})
    homology = EquivariantHomology(complex_, action)

    induced = homology.induced_homology_map(mapping, 1)
    representation = homology.representation_matrix(reflection, 1)

    assert induced == Matrix([[-1]])
    assert representation == Matrix([[-1]])
    assert representation * induced == induced * representation


def test_induced_homology_map_rejects_nonequivariant_self_map():
    complex_ = SimplicialComplex(
        {0, 1, 2},
        facet_set=[{0, 1}, {1, 2}, {0, 2}],
    )
    rotation = Permutation([1, 2, 0])
    action = PermutationAction((0, 1, 2), [rotation])
    transposition_map = SimplicialMap(complex_, complex_, {0: 1, 1: 0, 2: 2})
    homology = EquivariantHomology(complex_, action)

    with pytest.raises(ValueError, match="equivariant under each action generator"):
        homology.induced_homology_map(transposition_map, 1)


def test_reduced_induced_homology_map():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    swap = Permutation([1, 0])
    action = PermutationAction((0, 1), [swap])
    mapping = SimplicialMap(complex_, complex_, {0: 1, 1: 0})
    homology = EquivariantHomology(complex_, action, reduced=True)

    assert homology.induced_homology_map(mapping, 0) == Matrix([[-1]])


def test_induced_homology_map_rejects_different_complex():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    other = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    action = PermutationAction((0, 1), [Permutation([1, 0])])
    mapping = SimplicialMap(complex_, other, {0: 0, 1: 1})
    homology = EquivariantHomology(complex_, action)

    with pytest.raises(ValueError, match="self-map of the acted-on complex"):
        homology.induced_homology_map(mapping, 0)
