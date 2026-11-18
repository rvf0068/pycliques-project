"""Regression and certificate-integrity tests for issue #16.

GUDHI is the production backend. The integer-bitset matrix reduction in
bounded_betti is an independent exact GF(2) implementation for graph flag
complexes. Keep these checks even if the homology API later changes.
"""

from __future__ import annotations

import networkx as nx
import pytest
from pycliques.cliques import clique_graph
from pycliques.named import snub_disphenoid
from pycombtop.bounded_betti import bounded_reduced_betti_mod2
from pycombtop.homology_backend import HomologyResourceLimit, reduced_betti_facets_mod2
from pycombtop.homotopy_type import betti_numbers_graph, betti_numbers_sc
from pycombtop.simplex import Simplex, SimplicialComplex


def test_snub_iterates_independently_certified_gf2():
    graph = snub_disphenoid()
    for order, edges in ((8, 18), (12, 46), (20, 136)):
        assert graph.number_of_nodes() == order
        if order != 8:
            assert graph.number_of_edges() == edges
        expected = [0, 0, 1]
        assert betti_numbers_graph(graph) == expected
        binary = bounded_reduced_betti_mod2(graph, max_faces=150_000)
        assert binary is not None
        assert binary.get(0) == 0
        assert binary.get(1) == 0
        assert binary.get(2) == 1
        assert all(v == 0 for d, v in binary.items() if d != 2)
        if order != 20:
            next_graph = clique_graph(graph, bound=1200)
            assert next_graph is not None
            graph = next_graph


def test_nonflag_triangle_boundary_is_not_filled():
    # Its one-skeleton is K3. Expanding that graph would incorrectly
    # fill the 2-simplex and destroy H_1!
    sc = SimplicialComplex(
        {"a", "b", "c"},
        facet_set=[
            Simplex({"a", "b"}),
            Simplex({"b", "c"}),
            Simplex({"c", "a"}),
        ],
    )
    assert betti_numbers_sc(sc) == [0, 1]
    assert betti_numbers_graph(sc.one_skeleton_graph()) == []


def test_sphere_and_disconnected_reduced_homology():
    sphere = SimplicialComplex(
        {0, 1, 2, 3},
        facet_set=[{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}],
    )
    assert betti_numbers_sc(sphere) == [0, 0, 1]
    isolated = SimplicialComplex({"left", "right"}, facet_set=[{"left"}, {"right"}])
    assert betti_numbers_sc(isolated) == [1]
    assert betti_numbers_graph(nx.empty_graph(2)) == [1]
    assert betti_numbers_graph(nx.Graph()) == []


def test_explicit_limits_never_produce_fake_zero_betti():
    with pytest.raises(HomologyResourceLimit):
        reduced_betti_facets_mod2([{0, 1, 2}], max_faces=6)
    assert reduced_betti_facets_mod2([{0, 1, 2}], max_faces=7) == [0, 0, 0]
    with pytest.raises(ValueError):
        reduced_betti_facets_mod2([{0}], max_faces=0)
    with pytest.raises(ValueError):
        reduced_betti_facets_mod2([{0}], max_faces=True)


def test_exact_vertex_decomposability_refuses_incomplete_homology(monkeypatch):
    import pycombtop.homotopy_type as module

    def limited(_sc):
        raise HomologyResourceLimit("synthetic face-budget overflow")

    monkeypatch.setattr(module, "betti_numbers_sc", limited)
    simplex = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    # This branch can certify shellability but MUST NOT invent the wedge
    # multiplicity when the homology backend cannot finish.
    assert module._try_vertex_decomposable(simplex) is None
