"""Small exact regressions for finite posets and LPV08 graph bridges."""

from dataclasses import replace
from itertools import combinations

import networkx as nx
import pytest
from pycliques import clique_graph, homotopy_clique_graph
from pycliques.cliques import double_bound_graph as graph_double_bound
from pycliques.helly import is_clique_helly

from pycombtop import (
    BeatMove,
    Poset,
    SimplicialComplex,
    atomic_projection,
    atomic_subposet,
    beat_core,
    beat_moves,
    betti_numbers_sc,
    bounded_complete_poset,
    bounded_complex,
    clique_complex,
    coatomic_projection,
    coatomic_subposet,
    complete_poset,
    crosscut_complex,
    double_bound_graph,
    face_poset,
    interval_poset,
    is_atomized,
    is_coatomized,
    is_crosscut,
    is_down_helly,
    is_down_sperner,
    is_join_increasing,
    is_strongly_up_sperner,
    is_up_helly,
    is_up_sperner,
    join_contraction,
    lower_fibers,
    mho,
    nice_fibers,
    omega,
    order_complex,
    product_poset,
    verify_beat_reduction,
    verify_conical_contraction,
)


def test_order_construction_and_validation():
    p = Poset.from_covers(
        [None, "middle", 3, ("isolated",)], [(None, "middle"), ("middle", 3)]
    )
    assert p.leq(None, 3)
    assert not p.comparable(3, ("isolated",))
    assert p.upper_covers(None) == {"middle"}
    assert p.lower_covers(3) == {"middle"}
    assert p.lower_set(3, strict=True) == {None, "middle"}
    assert p.upper_set(None) == {None, "middle", 3}
    assert p.interval(None, 3) == {None, "middle", 3}
    assert not p.interval(3, None)
    assert p.induced([None, 3]).upper_covers(None) == {3}
    assert p.dual().dual() == p
    assert Poset.from_relation(p, p.relation()) == p
    assert p.join([None, "middle"]) == "middle"
    assert p.meet([None, 3]) is None
    assert p.is_ideal([None, "middle"])
    assert not p.is_ideal([3])
    assert not p.is_ideal(["unknown"])
    assert order_complex(p).dimension() == 2
    assert is_down_helly(p)
    with pytest.raises(KeyError):
        p.leq(None, "unknown")
    with pytest.raises(ValueError):
        p.induced(["unknown"])
    with pytest.raises(ValueError):
        p.join([3, ("isolated",)])
    with pytest.raises(ValueError):
        Poset([0, 1], [(0, 1), (1, 0)])
    with pytest.raises(ValueError):
        Poset([0], [(0, 0)])
    with pytest.raises(ValueError):
        Poset.from_relation([0, 1, 2], [(0, 1), (1, 2)])
    with pytest.raises(ValueError):
        Poset([0], [(0, 1)])
    with pytest.raises(ValueError):
        Poset.from_relation([0], [(1, 1)])
    assert Poset.from_order_matrix(["a", "b"], [[True, True], [False, True]]) == Poset(
        ["a", "b"], [("a", "b")]
    )
    for matrix in (
        [[True]],
        [[False, True], [False, True]],
        [[True, True], [True, True]],
        [[True, 2], [False, True]],
    ):
        with pytest.raises(ValueError):
            Poset.from_order_matrix([0, 1], matrix)


def test_beat_core_trace_and_maps():
    p = Poset(range(3), [(0, 1), (1, 2)])
    assert BeatMove(0, 1, "up") in beat_moves(p)
    assert BeatMove(2, 1, "down") in beat_moves(p)
    result = beat_core(p)
    assert len(result.core) == 1
    assert verify_beat_reduction(p, result)
    assert not verify_beat_reduction(
        p, replace(result, moves=(BeatMove(0, 2, "up"),))
    )
    assert not verify_beat_reduction(p, replace(result, moves=()))
    assert not verify_beat_reduction(p, replace(result, core=p))
    mapping = join_contraction(p, 1)
    assert verify_conical_contraction(p, 1, mapping)
    assert not verify_conical_contraction(p, 1, {0: 0, 1: 1, 2: 2})
    assert not p.is_order_preserving(p, {0: 2, 1: 1, 2: 0})
    assert not p.is_order_preserving(p, {0: 0})
    assert lower_fibers(p, p, mapping)[1] == p.induced([0, 1])
    with pytest.raises(ValueError):
        lower_fibers(p, p, {})
    with pytest.raises(ValueError):
        join_contraction(p, 3)


def test_face_poset_barycentric_subdivision_and_sphere_core():
    triangle = SimplicialComplex(
        [0, 1, 2, "ghost"], facet_set=[{0, 1}, {1, 2}, {0, 2}]
    )
    p = face_poset(triangle)
    assert len(p) == 6
    assert frozenset() not in p
    assert frozenset({"ghost"}) not in p
    sd = order_complex(p)
    assert len(sd.facet_set) == 6
    assert sd.dimension() == 1
    assert betti_numbers_sc(sd) == betti_numbers_sc(triangle) == [0, 1]
    assert not beat_core(p).moves
    assert verify_beat_reduction(p, beat_core(p))
    assert is_atomized(p)
    assert is_coatomized(p)
    assert is_strongly_up_sperner(p)
    assert is_join_increasing(p)
    assert not is_down_helly(p)  # nonflag triangle boundary
    assert not is_up_helly(p)
    assert is_crosscut(p, p.minimal_elements())
    assert crosscut_complex(p, p.minimal_elements()) == bounded_complex(p)
    assert not is_crosscut(p, p)
    assert not is_crosscut(p, [])
    assert not is_crosscut(p, ["unknown"])
    assert betti_numbers_sc(order_complex(bounded_complete_poset(p))) == [0, 1]
    assert betti_numbers_sc(order_complex(bounded_complete_poset(p, upper=False))) == [
        0,
        1,
    ]


@pytest.mark.parametrize(
    "graph",
    [nx.path_graph(3), nx.cycle_graph(4), nx.complete_graph(3), nx.empty_graph(2)],
)
def test_graph_bridges(graph):
    p = complete_poset(graph)
    g = omega(p)
    q = mho(p)
    kg = clique_graph(graph)
    assert kg is not None
    assert set(q) == set(kg)
    assert {frozenset(e) for e in q.edges()} == {frozenset(e) for e in kg.edges()}
    assert nx.is_isomorphic(g, graph)
    assert is_down_helly(p)
    assert is_up_helly(p) == is_clique_helly(graph)
    assert is_up_sperner(p)
    assert is_atomized(p)
    assert is_coatomized(p)
    assert is_join_increasing(p)
    assert is_strongly_up_sperner(p)
    assert atomic_subposet(p) == p
    h = coatomic_projection(p)
    assert p.is_order_preserving(p, h)
    assert all(p.leq(x, h[x]) and h[h[x]] == h[x] for x in p)
    assert coatomic_subposet(p) == p.induced(h.values())
    assert betti_numbers_sc(order_complex(p)) == betti_numbers_sc(
        clique_complex(graph)
    )
    intervals = interval_poset(p)
    assert nx.is_isomorphic(omega(intervals), double_bound_graph(p))
    db = graph_double_bound(graph)
    assert db is not None
    assert set(db) == set(double_bound_graph(p))
    assert {frozenset(e) for e in db.edges()} == {
        frozenset(e) for e in double_bound_graph(p).edges()
    }
    hg = homotopy_clique_graph(graph)
    mapped = nx.relabel_nodes(mho(intervals), lambda i: (next(iter(i[0])), i[1]))
    assert set(mapped) == set(hg)
    assert {frozenset(e) for e in mapped.edges()} == {frozenset(e) for e in hg.edges()}
    assert betti_numbers_sc(order_complex(intervals)) == betti_numbers_sc(
        order_complex(p)
    )


def test_non_atomized_and_non_join_increasing():
    p = Poset("abUV", [(a, b) for a in "ab" for b in "UV"])
    assert not is_atomized(p)
    assert not is_coatomized(p)
    assert not is_up_sperner(p)
    assert not is_down_sperner(p)
    assert not is_strongly_up_sperner(p)
    assert not is_join_increasing(p)
    assert not is_crosscut(p, "ab")
    with pytest.raises(ValueError):
        atomic_projection(p)
    with pytest.raises(ValueError):
        join_contraction(p, "a")
    # Three minima with the same join for every pair.
    q = Poset("abcT", [(a, "T") for a in "abc"])
    assert is_atomized(q) and is_strongly_up_sperner(q)
    assert not is_join_increasing(q)
    # Compare the optimized recognition against direct subset enumeration.
    for candidate in (p, q, p.dual(), q.dual(), interval_poset(q)):
        minima = tuple(candidate.minimal_elements())
        bounded = [
            face
            for size in range(1, len(minima) + 1)
            for face in combinations(minima, size)
            if candidate.upper_bounds(face)
        ]
        try:
            joins = {frozenset(face): candidate.join(face) for face in bounded}
        except ValueError:
            assert not is_atomized(candidate)
        else:
            assert is_atomized(candidate)
            expected = all(
                not x < y or joins[x] != joins[y] for x in joins for y in joins
            )
            assert is_join_increasing(candidate) == expected


def test_product_intervals_nice_fibers_and_empty():
    chain = Poset([0, 1], [(0, 1)])
    p = product_poset(chain, chain)
    assert len(p) == 4
    assert p.leq((0, 0), (1, 1))
    assert not p.comparable((1, 0), (0, 1))
    assert p.is_ideal([(0, 0), (0, 1)])
    assert interval_poset(chain) == Poset(
        [(0, 0), (1, 1), (0, 1)],
        [((0, 0), (0, 1)), ((1, 1), (0, 1))],
    )
    fibers = nice_fibers(chain, chain.induced([0]))
    assert all(len(fiber) == 1 for fiber in fibers.values())
    with pytest.raises(ValueError):
        nice_fibers(chain, Poset([0, 1]))
    assert not nice_fibers(chain, Poset([]))[0].elements
    empty = Poset([])
    assert not beat_core(empty).moves
    assert verify_beat_reduction(empty, beat_core(empty))
    assert order_complex(empty).dimension() == -1
    assert face_poset(order_complex(empty)) == empty
    assert not omega(empty) and not mho(empty)
    assert not len(interval_poset(empty))
    assert not len(product_poset(empty, chain))
