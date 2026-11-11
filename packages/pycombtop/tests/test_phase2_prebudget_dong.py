"""Phase-2 regressions: safe budget recovery and reproducible Dong matchings."""

import random

import networkx as nx

from pycombtop import clique_complex, dong_matching
from pycombtop.homotopy_type import (
    _graph_exceeds_simplex_budget,
    _simplify_graph_ht,
    _try_dong,
    homotopy_type_with_verdict,
)


def test_prebudget_s_collapse_certifies_bounded_noncone():
    # A pendant edge attached to C4 makes an 11-face complex; the
    # s-dismantlable pendant vertex reduces it to C4, which has 9 faces.
    g = nx.cycle_graph(4)
    g.add_edge(0, 4)
    assert _graph_exceeds_simplex_budget(g, 9)
    core = _simplify_graph_ht(g)
    assert nx.is_isomorphic(core, nx.cycle_graph(4))
    assert not _graph_exceeds_simplex_budget(core, 9)
    verdict = homotopy_type_with_verdict(g, max_simplices=9)
    assert verdict.is_exact and not verdict.is_limited
    assert verdict.wedge.spheres == {1: 1}


def test_strict_vertex_budget_survives_prebudget_reduction():
    g = nx.cycle_graph(4)
    g.add_edge(0, 4)
    verdict = homotopy_type_with_verdict(
        g, max_vertices=4, max_simplices=9,
    )
    assert verdict.is_limited and not verdict.is_exact


def test_no_invalid_type_from_irreducible_overbudget_core():
    g = nx.cycle_graph(5)
    assert nx.is_isomorphic(_simplify_graph_ht(g), g)
    verdict = homotopy_type_with_verdict(g, max_simplices=10)
    assert verdict.is_limited and not verdict.is_exact


def test_dong_face_pairs_ignore_vertex_insertion_order():
    names = ("alpha", "beta", "gamma", "delta", "epsilon")
    edges = [(names[i], names[(i + 1) % 5]) for i in range(5)]
    g1 = nx.Graph()
    g1.add_nodes_from(names)
    g1.add_edges_from(edges)
    g2 = nx.Graph()
    g2.add_nodes_from(reversed(names))
    g2.add_edges_from(reversed(edges))
    sc1, sc2 = clique_complex(g1), clique_complex(g2)
    assert dong_matching(sc1, return_pairs=True) == dong_matching(
        sc2, return_pairs=True,
    )
    assert _try_dong(sc1, seed=19).wedge.spheres == {1: 1}
    assert _try_dong(sc2, seed=19).wedge.spheres == {1: 1}


def test_dong_seed_is_local_and_reproducible():
    # The API must not advance global random state, even when six shuffled
    # vertex orderings are attempted.
    g = nx.cycle_graph(5)
    state = random.getstate()
    first = homotopy_type_with_verdict(g, dong_seed=12345)
    assert random.getstate() == state
    second = homotopy_type_with_verdict(g, dong_seed=12345)
    assert (first.is_exact, first.wedge, first.reason) == (
        second.is_exact, second.wedge, second.reason
    )
