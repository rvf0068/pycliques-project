"""Tests for exact equivariant homology."""

import networkx as nx
from pycombtop import SimplicialComplex, clique_complex
from pyhomrep import (
    OrbitChainComplex,
    YoungTableaux,
    bouc_decomposition,
    decompose_young_homology,
    decompose_young_homology_map,
    matching_graph,
    partitions_list,
)


def test_reversing_stabilizer():
    sc = clique_complex(nx.complete_graph(2))
    chain = OrbitChainComplex(sc, [lambda v: 1 - v], reduced=True)
    assert chain.chain_dimension(0) == 1
    assert chain.chain_dimension(1) == 0
    assert chain.betti_number(0) == 0


def test_bouc_small():
    for n, k in [(4, 0), (5, 1), (6, 1)]:
        sc = clique_complex(matching_graph(n))

        def action(g, v):
            return tuple(sorted(g(i) if i < g.size else i for i in v))

        assert decompose_young_homology(
            sc, n, k, action, reduced=True
        ) == bouc_decomposition(n, k)


def test_homology_map():
    sc = clique_complex(nx.cycle_graph(4))
    tgt = clique_complex(nx.complete_graph(4))

    # A trivial S_2 action on both complexes.
    def action(g, v):
        return v
    result = decompose_young_homology_map(sc, tgt, 2, 1, action, action, lambda v: v)
    assert result == {"kernel": {(2,): 1}, "image": {}, "cokernel": {}}


def test_rim_hook_characters():
    for n in range(1, 6):
        for lam in partitions_list(n):
            for rho in partitions_list(n):
                yt = YoungTableaux(lam, rho)
                assert yt.CMNR() == sum((-1) ** h for h in yt.heights())


def test_relative_orientation():
    sc = clique_complex(nx.complete_graph(2))
    points = SimplicialComplex({0, 1}, facet_set=[{0}, {1}])
    assert OrbitChainComplex(sc, relative=points).betti_number(1) == 1
    chain = OrbitChainComplex(sc, [lambda v: 1 - v], relative=points)
    assert chain.betti_number(1) == 0


def test_identity_homology_map():
    sc = clique_complex(nx.cycle_graph(4))

    def action(g, v):
        return v

    result = decompose_young_homology_map(sc, sc, 2, 1, action, action, lambda v: v)
    assert result == {"kernel": {}, "image": {(2,): 1}, "cokernel": {}}
