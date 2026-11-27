"""Standard operations on finite complexes, including ambient labels."""

from itertools import combinations, product

import networkx as nx
import pytest
from pycombtop import (
    SimplicialComplex,
    alexander_dual,
    barycentric_subdivision,
    betti_numbers_sc,
    cone,
    face_deletion,
    face_poset,
    induced_subcomplex,
    minimal_nonfaces,
    order_complex,
    relabel_complex,
    simplex_boundary,
    simplicial_join,
    sphere_complex,
    suspension,
    wedge_sum,
)


def _powerset(ground):
    return [
        frozenset(s) for n in range(len(ground) + 1) for s in combinations(ground, n)
    ]


def test_operations_preserve_base_set_and_do_not_mutate_input():
    x = SimplicialComplex({0, 1, "ghost"}, [{0, 1}])
    y = induced_subcomplex(x, {1, "ghost"})
    assert y.base_set == {1, "ghost"}
    assert y.vertex_set == {1}
    assert relabel_complex(x, {0: "a", 1: "b", "ghost": "c"}) == SimplicialComplex(
        {"a", "b", "c"}, [{"a", "b"}]
    )
    assert x == SimplicialComplex({0, 1, "ghost"}, [{0, 1}])
    with pytest.raises(ValueError, match="outside"):
        induced_subcomplex(x, {2})
    with pytest.raises(ValueError, match="injective"):
        relabel_complex(x, {0: 1, 1: 1, "ghost": 2})
    with pytest.raises(ValueError, match="entire"):
        relabel_complex(x, {0: "a", 1: "b"})


def test_join_cone_suspension_and_wedge():
    x = sphere_complex(1)
    y = relabel_complex(sphere_complex(0), {0: "a", 1: "b"})
    j = simplicial_join(x, y)
    assert j.dimension() == 2
    assert betti_numbers_sc(j) == [0, 0, 1]
    assert suspension(x, ("a", "b")) == j
    assert suspension(x).dimension() == 2
    c = cone(x, "apex")
    assert c.dimension() == 2
    assert all("apex" in f for f in c.facet_set)
    assert all(b == 0 for b in betti_numbers_sc(cone(x)))
    w = wedge_sum(x, y, 0, "a")
    assert w.vertex_set == {0, 1, 2, "b"}
    assert betti_numbers_sc(w) == [1, 1]
    w2 = wedge_sum(x, relabel_complex(x, {0: 0, 1: "p", 2: "q"}), 0, 0)
    assert betti_numbers_sc(w2) == [0, 2]
    with pytest.raises(ValueError, match="disjoint"):
        simplicial_join(x, x)
    with pytest.raises(ValueError, match="disjoint"):
        cone(x, 0)
    with pytest.raises(ValueError, match="distinct"):
        suspension(x, ("a", "a"))
    with pytest.raises(ValueError, match="basepoints"):
        wedge_sum(x, y, 7, "a")
    with pytest.raises(ValueError, match="disjoint"):
        wedge_sum(x, x, 0, 0)


def test_face_deletion():
    x = SimplicialComplex(range(5), [{0, 1, 2}, {0, 1, 3}, {2, 4}])
    d = face_deletion(x, {0, 1})
    assert d.base_set == x.base_set
    assert d.all_simplices() == {
        f for f in x.all_simplices() if not {0, 1} <= f
    }
    assert face_deletion(x, {1, 4}) == x
    assert face_deletion(x, set()).facet_set == set()
    assert face_deletion(x, {0}).vertex_set == x.deletion(0).vertex_set
    with pytest.raises(ValueError, match="outside"):
        face_deletion(x, {9})


def test_barycentric_subdivision_nonflag_and_ghosts():
    x = SimplicialComplex(range(4), [{0, 1}, {0, 2}, {1, 2}])
    sd = barycentric_subdivision(x)
    assert sd == order_complex(face_poset(x))
    assert nx.is_isomorphic(sd.one_skeleton_graph(), nx.cycle_graph(6))
    assert sd.is_clique_complex()
    assert betti_numbers_sc(sd) == [0, 1]
    assert len(barycentric_subdivision(simplex_boundary(4)).vertex_set) == 14


def test_void_and_empty_face_conventions():
    void = SimplicialComplex(set(), [])
    unit = SimplicialComplex(set(), [set()])
    x = sphere_complex(1)
    assert simplicial_join(unit, x) == x
    assert simplicial_join(void, x).facet_set == set()
    assert barycentric_subdivision(void) == void
    assert barycentric_subdivision(unit) == unit
    assert cone(unit).dimension() == 0
    assert cone(void).facet_set == set()
    assert minimal_nonfaces(void) == {frozenset()}
    assert not minimal_nonfaces(unit)
    assert alexander_dual(void) == unit
    assert alexander_dual(unit) == void


def test_duality_exhaustively_on_three_point_base():
    ground = set(range(3))
    subsets = _powerset(ground)
    for selected in product((False, True), repeat=len(subsets)):
        x = SimplicialComplex(
            ground, [f for f, include in zip(subsets, selected, strict=True) if include]
        )
        faces = x.all_simplices()
        expected = {
            f for f in subsets if f not in faces and all(f - {v} in faces for v in f)
        }
        assert minimal_nonfaces(x) == expected
        dual = alexander_dual(x)
        assert dual.base_set == ground
        assert dual.all_simplices() == {
            f for f in subsets if frozenset(ground - f) not in faces
        }
        assert alexander_dual(dual) == x
