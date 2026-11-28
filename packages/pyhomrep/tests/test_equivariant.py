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


def test_induced_homology_map_validates_type_and_degree():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    action = PermutationAction((0, 1), [])
    mapping = SimplicialMap(complex_, complex_, {0: 0, 1: 1})
    homology = EquivariantHomology(complex_, action)

    with pytest.raises(TypeError, match="must be a SimplicialMap"):
        homology.induced_homology_map(None, 0)
    with pytest.raises(ValueError, match="degree must be"):
        homology.induced_homology_map(mapping, -1)
    with pytest.raises(ValueError, match="degree must be"):
        homology.induced_homology_map(mapping, True)


def test_induced_homology_map_returns_empty_matrix_outside_homology_range():
    complex_ = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    action = PermutationAction((0, 1, 2), [])
    mapping = SimplicialMap(complex_, complex_, {0: 0, 1: 1, 2: 2})
    homology = EquivariantHomology(complex_, action)

    assert homology.induced_homology_map(mapping, 1) == Matrix(0, 0, [])
    assert homology.induced_homology_map(mapping, 3) == Matrix(0, 0, [])


def test_induced_homology_map_between_differently_labeled_actions():
    source = SimplicialComplex(
        ("a", "b", "c", "d"),
        facet_set=[{"a", "b"}, {"b", "c"}, {"c", "d"}, {"a", "d"}],
    )
    target = SimplicialComplex(
        ("w", "x", "y", "z"),
        facet_set=[{"w", "x"}, {"x", "y"}, {"y", "z"}, {"w", "z"}],
    )
    rotation = Permutation([1, 2, 3, 0])
    inverse_rotation = Permutation([3, 0, 1, 2])
    source_action = PermutationAction(("a", "b", "c", "d"), [rotation])
    # A different generating set for the same concrete subgroup of S_4.
    target_action = PermutationAction(("w", "x", "y", "z"), [inverse_rotation])
    mapping = SimplicialMap(source, target, {"a": "w", "b": "x", "c": "y", "d": "z"})
    source_homology = EquivariantHomology(source, source_action)
    target_homology = EquivariantHomology(target, target_action)

    induced = source_homology.induced_homology_map_to(target_homology, mapping, 1)
    source_representation = source_homology.representation_matrix(rotation, 1)
    target_representation = target_homology.representation_matrix(rotation, 1)

    assert induced == Matrix([[1]])
    assert target_representation * induced == induced * source_representation


def test_induced_homology_map_to_rejects_different_permutation_groups():
    source = SimplicialComplex(
        {0, 1, 2, 3},
        facet_set=[{0, 1}, {1, 2}, {2, 3}, {0, 3}],
    )
    target = SimplicialComplex(
        {"a", "b", "c", "d"},
        facet_set=[{"a", "b"}, {"b", "c"}, {"c", "d"}, {"a", "d"}],
    )
    rotation = Permutation([1, 2, 3, 0])
    reflection = Permutation([0, 3, 2, 1])
    source_action = PermutationAction((0, 1, 2, 3), [rotation])
    target_action = PermutationAction(("a", "b", "c", "d"), [reflection])
    mapping = SimplicialMap(source, target, {0: "a", 1: "b", 2: "c", 3: "d"})
    source_homology = EquivariantHomology(source, source_action)
    target_homology = EquivariantHomology(target, target_action)

    with pytest.raises(ValueError, match="subgroup of the target action"):
        source_homology.induced_homology_map_to(target_homology, mapping, 1)


def test_induced_homology_map_to_rejects_incompatible_reduced_conventions():
    source = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    target = SimplicialComplex({"a", "b"}, facet_set=[{"a"}, {"b"}])
    swap = Permutation([1, 0])
    source_action = PermutationAction((0, 1), [swap])
    target_action = PermutationAction(("a", "b"), [swap])
    mapping = SimplicialMap(source, target, {0: "a", 1: "b"})
    source_homology = EquivariantHomology(source, source_action)
    target_homology = EquivariantHomology(target, target_action, reduced=True)

    with pytest.raises(ValueError, match="agree on reduced homology"):
        source_homology.induced_homology_map_to(target_homology, mapping, 0)


def test_induced_homology_map_to_can_be_rectangular():
    source = SimplicialComplex(
        ("a", "b", "c", "d"),
        facet_set=[{"a", "b"}, {"b", "c"}, {"c", "d"}, {"a", "d"}],
    )
    target = SimplicialComplex(
        ("w", "x", "y", "z"),
        facet_set=[{"w", "x"}, {"x", "y"}, {"y", "z"}],
    )
    source_action = PermutationAction(("a", "b", "c", "d"), [])
    target_action = PermutationAction(("w", "x", "y", "z"), [])
    mapping = SimplicialMap(source, target, {"a": "w", "b": "x", "c": "y", "d": "x"})
    source_homology = EquivariantHomology(source, source_action)
    target_homology = EquivariantHomology(target, target_action)

    induced = source_homology.induced_homology_map_to(target_homology, mapping, 1)

    assert induced == Matrix(0, 1, [])


def test_induced_homology_map_to_validates_target_and_map_types():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    action = PermutationAction((0, 1), [])
    homology = EquivariantHomology(complex_, action)
    mapping = SimplicialMap(complex_, complex_, {0: 0, 1: 1})

    with pytest.raises(TypeError, match="target must be"):
        homology.induced_homology_map_to(None, mapping, 0)
    with pytest.raises(TypeError, match="simplicial_map must be"):
        homology.induced_homology_map_to(homology, None, 0)


def test_induced_homology_map_to_validates_map_complexes_and_action_sizes():
    source = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    target = SimplicialComplex({"a", "b"}, facet_set=[{"a"}, {"b"}])
    source_action = PermutationAction((0, 1), [])
    target_action = PermutationAction(("a", "b"), [])
    source_homology = EquivariantHomology(source, source_action)
    target_homology = EquivariantHomology(target, target_action)
    wrong_source = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    wrong_map = SimplicialMap(wrong_source, target, {0: "a", 1: "a"})

    with pytest.raises(ValueError, match="source and target must match"):
        source_homology.induced_homology_map_to(target_homology, wrong_map, 0)

    larger_target = SimplicialComplex({"a", "b", "c"}, facet_set=[{"a"}, {"b"}, {"c"}])
    larger_action = PermutationAction(("a", "b", "c"), [])
    larger_homology = EquivariantHomology(larger_target, larger_action)
    map_to_larger = SimplicialMap(source, larger_target, {0: "a", 1: "b"})
    with pytest.raises(ValueError, match="same number of ordered positions"):
        source_homology.induced_homology_map_to(larger_homology, map_to_larger, 0)


def test_induced_homology_map_to_checks_equivariance_and_degree():
    complex_ = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}, {1, 2}, {0, 2}])
    rotation = Permutation([1, 2, 0])
    action = PermutationAction((0, 1, 2), [rotation])
    homology = EquivariantHomology(complex_, action)
    non_equivariant = SimplicialMap(complex_, complex_, {0: 1, 1: 0, 2: 2})

    with pytest.raises(ValueError, match="intertwine"):
        homology.induced_homology_map_to(homology, non_equivariant, 1)
    with pytest.raises(ValueError, match="degree must be"):
        homology.induced_homology_map_to(
            homology,
            SimplicialMap(complex_, complex_, {0: 0, 1: 1, 2: 2}),
            True,
        )
    with pytest.raises(ValueError, match="degree must be"):
        homology.induced_homology_map_to(
            homology,
            SimplicialMap(complex_, complex_, {0: 0, 1: 1, 2: 2}),
            -1,
        )


def test_induced_homology_map_to_returns_empty_matrix_when_homology_vanishes():
    source = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    target = SimplicialComplex({"a", "b"}, facet_set=[{"a"}, {"b"}])
    source_action = PermutationAction((0, 1), [])
    target_action = PermutationAction(("a", "b"), [])
    source_homology = EquivariantHomology(source, source_action)
    target_homology = EquivariantHomology(target, target_action)
    mapping = SimplicialMap(source, target, {0: "a", 1: "a"})

    induced = source_homology.induced_homology_map_to(target_homology, mapping, 1)

    assert induced == Matrix(0, 0, [])


def test_induced_homology_map_to_restricts_a_larger_target_action():
    source = SimplicialComplex(
        ("a", "b", "c"),
        facet_set=[{"a", "b"}, {"b", "c"}, {"a", "c"}],
    )
    target = SimplicialComplex(
        ("x", "y", "z"),
        facet_set=[{"x", "y"}, {"y", "z"}, {"x", "z"}],
    )
    transposition = Permutation([1, 0, 2])
    three_cycle = Permutation([1, 2, 0])
    source_action = PermutationAction(("a", "b", "c"), [transposition])
    target_action = PermutationAction(("x", "y", "z"), [transposition, three_cycle])
    mapping = SimplicialMap(source, target, {"a": "x", "b": "y", "c": "z"})
    source_homology = EquivariantHomology(source, source_action)
    target_homology = EquivariantHomology(target, target_action)

    induced = source_homology.induced_homology_map_to(target_homology, mapping, 1)
    source_representation = source_homology.representation_matrix(transposition, 1)
    target_representation = target_homology.representation_matrix(transposition, 1)

    assert target_action.group.order() == 6
    assert source_action.group.order() == 2
    assert induced == Matrix([[1]])
    assert target_representation * induced == induced * source_representation


def test_relative_equivariant_homology_of_interval_mod_boundary():
    complex_ = SimplicialComplex((0, 1), facet_set=[{0, 1}])
    boundary = SimplicialComplex((0, 1), facet_set=[{0}, {1}])
    swap = Permutation([1, 0])
    action = PermutationAction((0, 1), [swap])
    homology = EquivariantHomology(complex_, action, relative_to=boundary)

    assert homology.character(swap, 1) == -1
    assert homology.character(swap, 0) == 0
    assert homology.representation_matrix(swap, 1) == Matrix([[-1]])


def test_reduced_relative_homology_does_not_augment_nonempty_subcomplex():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    relative = SimplicialComplex({0}, facet_set=[{0}])
    action = PermutationAction((0, 1), [])
    homology = EquivariantHomology(complex_, action, reduced=True, relative_to=relative)

    # H_0(X, A) has rank one; an erroneous augmentation would kill it.
    assert homology.representation_matrix(Permutation(size=2), 0) == Matrix([[1]])


def test_relative_equivariant_homology_rejects_noninvariant_subcomplex():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    relative = SimplicialComplex({0}, facet_set=[{0}])
    swap = Permutation([1, 0])
    action = PermutationAction((0, 1), [swap])

    with pytest.raises(ValueError, match="relative complex must be invariant"):
        EquivariantHomology(complex_, action, relative_to=relative)


def test_relative_equivariant_homology_rejects_non_subcomplex():
    complex_ = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    relative = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    action = PermutationAction((0, 1), [])

    with pytest.raises(ValueError, match="relative complex must be a subcomplex"):
        EquivariantHomology(complex_, action, relative_to=relative)


def test_induced_relative_homology_map_requires_map_of_pairs():
    complex_ = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}, {1, 2}])
    relative = SimplicialComplex({0, 2}, facet_set=[{0}, {2}])
    action = PermutationAction((0, 1, 2), [])
    homology = EquivariantHomology(
        complex_, action, relative_to=relative
    )
    mapping = SimplicialMap(complex_, complex_, {0: 1, 1: 1, 2: 2})

    with pytest.raises(ValueError, match="send the relative subcomplex"):
        homology.induced_homology_map(mapping, 0)


def test_induced_homology_map_between_relative_pairs():
    source = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    source_relative = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    target = SimplicialComplex({"a", "b"}, facet_set=[{"a", "b"}])
    target_relative = SimplicialComplex({"a", "b"}, facet_set=[{"a"}, {"b"}])
    swap = Permutation([1, 0])
    source_action = PermutationAction((0, 1), [swap])
    target_action = PermutationAction(("a", "b"), [swap])
    mapping = SimplicialMap(source, target, {0: "a", 1: "b"})
    source_homology = EquivariantHomology(
        source, source_action, relative_to=source_relative
    )
    target_homology = EquivariantHomology(
        target, target_action, relative_to=target_relative
    )

    induced = source_homology.induced_homology_map_to(target_homology, mapping, 1)
    assert induced == Matrix([[1]])
    assert target_homology.representation_matrix(swap, 1) * induced == (
        induced * source_homology.representation_matrix(swap, 1)
    )


def test_relative_equivariant_homology_validates_relative_argument_type():
    complex_ = SimplicialComplex({0}, facet_set=[{0}])
    action = PermutationAction((0,), [])

    with pytest.raises(TypeError, match="relative_to must be"):
        EquivariantHomology(complex_, action, relative_to=object())
