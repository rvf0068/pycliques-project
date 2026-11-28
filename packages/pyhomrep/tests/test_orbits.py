"""Tests for exact equivariant homology."""

import networkx as nx
import pytest
from pycombtop import PermutationAction, SimplicialComplex, clique_complex
from pyhomrep import (
    OrbitChainComplex,
    YoungTableaux,
    bouc_decomposition,
    decompose_induced_character,
    decompose_young_homology,
    decompose_young_homology_map,
    matching_graph,
    partitions_list,
)
from sympy.combinatorics import Permutation


def test_reversing_stabilizer():
    sc = clique_complex(nx.complete_graph(2))
    chain = OrbitChainComplex(sc, [lambda v: 1 - v], reduced=True)
    assert chain.chain_dimension(0) == 1
    assert chain.chain_dimension(1) == 0
    assert chain.betti_number(0) == 0


def test_bouc_small():
    for n, k in [(4, 0), (5, 1), (6, 1), (7, 2)]:
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


def test_octahedral_sphere_induction():
    # Signed cycle counts for the order-48 stabilizer of three centered
    # triples in S_9, acting on the 3-sphere Delta(K(O_3)).
    counts = {
        (6, 3): 8,
        (4, 2, 2, 1): -6,
        (4, 2, 1, 1, 1): -6,
        (3, 3, 3): 8,
        (2, 2, 2, 2, 1): -6,
        (2, 2, 2, 1, 1, 1): -5,
        (2, 2, 1, 1, 1, 1, 1): 3,
        (2, 1, 1, 1, 1, 1, 1, 1): 3,
        (1,) * 9: 1,
    }
    mult = decompose_induced_character(9, counts, 48)
    assert (5, 1, 1, 1, 1) not in mult
    assert mult[(3, 3, 3)] == 3
    assert (
        sum(m * YoungTableaux(list(lam), [1] * 9).CMNR() for lam, m in mult.items())
        == 7560
    )


def test_orbit_chain_complex_accepts_permutation_action():
    complex_ = clique_complex(nx.cycle_graph(3))
    action = PermutationAction(
        (0, 1, 2), [Permutation([1, 2, 0])]
    )
    chains = OrbitChainComplex(complex_, action, reduced=True)
    assert chains.betti_number(0) == 0
    assert chains.betti_number(1) == 0
    assert chains.betti_number(2) == 0


def test_orbit_chain_complex_rejects_noninvariant_relative_subcomplex():
    complex_ = clique_complex(nx.cycle_graph(3))
    action = PermutationAction(
        (0, 1, 2), [Permutation([1, 2, 0])]
    )
    relative = SimplicialComplex({0}, facet_set=[{0}])
    with pytest.raises(ValueError, match="must be invariant"):
        OrbitChainComplex(complex_, action, relative=relative)
