from sympy.combinatorics import Permutation
import pytest

from pycombtop import (
    PermutationAction,
    Poset,
    SimplicialComplex,
    boolean_lattice,
)


def test_permutation_action_on_labeled_set():
    action = PermutationAction(
        ("a", "b", "c"),
        [Permutation([1, 0, 2]), Permutation([1, 2, 0])],
    )

    assert action.group.order() == 6
    assert action.orbit("a") == ("a", "b", "c")
    assert action.orbits() == (("a", "b", "c"),)
    assert action.fixed_points() == ()
    assert action.stabilizer("a").order() == 2
    assert action.image("a", Permutation([0, 2, 1])) == "a"
    assert action.is_invariant({"a", "b", "c"})
    assert not action.is_invariant({"a"})


def test_trivial_action_and_empty_generators():
    action = PermutationAction(("x", "y"), [])
    assert action.group.order() == 1
    assert action.orbits() == (("x",), ("y",))
    assert action.fixed_points() == ("x", "y")


def test_action_validates_simplicial_automorphisms():
    complex_ = SimplicialComplex(
        {"a", "b", "c"}, facet_set=[{"a", "b"}, {"b", "c"}, {"a", "c"}]
    )
    action = PermutationAction(
        ("a", "b", "c"), [Permutation([1, 2, 0])]
    )
    action.validate_complex(complex_)

    invalid = PermutationAction(("a", "b", "c"), [Permutation([1, 0, 2])])
    path = SimplicialComplex({"a", "b", "c"}, facet_set=[{"a", "b"}, {"b", "c"}])
    with pytest.raises(ValueError, match="preserve the simplicial complex"):
        invalid.validate_complex(path)


def test_action_validates_poset_automorphisms():
    poset = boolean_lattice(2)
    elements = poset.elements
    index = {x: i for i, x in enumerate(elements)}
    def swap(x):
        return frozenset(1 - v for v in x)

    generator = Permutation([index[swap(x)] for x in elements], size=len(elements))
    PermutationAction(elements, [generator]).validate_poset(poset)

    chain = Poset((0, 1, 2), ((0, 1), (1, 2)))
    invalid = PermutationAction(chain.elements, [Permutation([2, 1, 0])])
    with pytest.raises(ValueError, match="poset automorphism"):
        invalid.validate_poset(chain)


def test_invariant_subcomplex_validation():
    ambient = SimplicialComplex(
        {0, 1, 2}, facet_set=[{0, 1}, {1, 2}, {0, 2}]
    )
    action = PermutationAction((0, 1, 2), [Permutation([1, 2, 0])])
    invariant = SimplicialComplex({0, 1, 2}, facet_set=[{0}, {1}, {2}])
    action.validate_invariant_subcomplex(ambient, invariant)

    noninvariant = SimplicialComplex({0}, facet_set=[{0}])
    with pytest.raises(ValueError, match="must be invariant"):
        action.validate_invariant_subcomplex(ambient, noninvariant)
