"""Exact multi-facet wedge certification and adversarial replay tests."""

from dataclasses import replace

import networkx as nx
import pytest
from pycombtop import (
    Simplex,
    SimplicialComplex,
    clique_complex,
    facet_puncture_wedge_certificate,
    verify_facet_puncture_wedge,
)


def test_four_cycle_is_one_circle():
    graph = nx.cycle_graph(4)
    source = clique_complex(graph)
    witness = facet_puncture_wedge_certificate(source)
    assert witness is not None
    assert witness.spheres == {1: 1}
    assert verify_facet_puncture_wedge(source, witness)


def test_o3_is_two_sphere():
    graph = nx.complete_multipartite_graph(2, 2, 2)
    source = clique_complex(graph)
    witness = facet_puncture_wedge_certificate(source)
    assert witness is not None
    assert witness.spheres == {2: 1}
    assert verify_facet_puncture_wedge(source, witness)
    assert not verify_facet_puncture_wedge(source, replace(witness, dimension=3))


def test_octagonal_prism_complement_is_three_sphere_wedge():
    """The same 16-vertex core appears in both nine-sphere K^3 iterates."""
    graph = nx.complement(nx.circular_ladder_graph(8))
    source = clique_complex(graph)
    witness = facet_puncture_wedge_certificate(source)
    assert witness is not None
    assert witness.spheres == {3: 3}
    assert len(witness.removed_facets) == 3
    # 1,154 nonempty simplices, three removed facets and one final vertex.
    assert len(witness.initial_pairs) + len(witness.final_pairs) == 575
    assert verify_facet_puncture_wedge(source, witness)
    assert witness == facet_puncture_wedge_certificate(source)

    # The finite enumeration budget must not lead to a false proof.
    assert facet_puncture_wedge_certificate(source, max_candidates=1) is None
    assert facet_puncture_wedge_certificate(source, max_faces=100) is None

    assert not verify_facet_puncture_wedge(
        source,
        replace(witness, removed_facets=(witness.removed_facets[0],) * 3),
    )
    assert not verify_facet_puncture_wedge(
        source,
        replace(witness, removed_facets=(*witness.removed_facets, Simplex({0}))),
    )
    assert not verify_facet_puncture_wedge(
        source,
        replace(witness, final_pairs=()),
    )
    corrupted_pair = (
        (witness.initial_pairs[0][1], witness.initial_pairs[0][1]),
        *witness.initial_pairs[1:],
    )
    assert not verify_facet_puncture_wedge(
        source, replace(witness, initial_pairs=corrupted_pair)
    )


def test_replay_rejects_forged_faces_and_honors_face_budgets():
    """A list with duplicated vertices can alias a different bitmask."""
    source = clique_complex(nx.cycle_graph(4))
    proof = facet_puncture_wedge_certificate(source)
    assert proof is not None
    # This would encode as 1+1=2 without requiring a genuine Simplex.
    altered = list(proof.final_pairs)
    altered[0] = ([0, 0], altered[0][1])
    assert not verify_facet_puncture_wedge(
        source, replace(proof, final_pairs=tuple(altered))
    )

    assert not verify_facet_puncture_wedge(
        source, replace(proof, removed_facets=([0, 1],))
    )

    # Four vertices and four edges: no hidden 500,000-face verifier limit.
    assert verify_facet_puncture_wedge(source, proof, max_faces=8)
    assert not verify_facet_puncture_wedge(source, proof, max_faces=7)
    assert verify_facet_puncture_wedge(source, proof)
    assert facet_puncture_wedge_certificate(source, max_faces=8) is not None


def test_input_shape_limits_and_no_false_certificates():
    empty = SimplicialComplex(set(), facet_set=[])
    assert facet_puncture_wedge_certificate(empty) is None
    assert not verify_facet_puncture_wedge(empty, object())

    # The bit-mask engine deliberately fails closed beyond 64 vertices.
    large_discrete = clique_complex(nx.empty_graph(65))
    assert facet_puncture_wedge_certificate(large_discrete) is None

    # The homological lower bound requires three deleted top facets.
    prism = clique_complex(nx.complement(nx.circular_ladder_graph(8)))
    assert facet_puncture_wedge_certificate(prism, max_removed_facets=2) is None

    # Reject removed facets that are nonmaximal, absent, or wrong-sized.
    circle = clique_complex(nx.cycle_graph(4))
    proof = facet_puncture_wedge_certificate(circle)
    assert proof is not None
    assert not verify_facet_puncture_wedge(
        circle, replace(proof, removed_facets=(Simplex({0}),))
    )
    assert not verify_facet_puncture_wedge(
        circle, replace(proof, removed_facets=(Simplex({0, 2}),))
    )
    assert not verify_facet_puncture_wedge(
        circle, replace(proof, initial_pairs=((Simplex({0}), Simplex({1, 2})),))
    )
    assert not verify_facet_puncture_wedge(
        circle, replace(proof, final_pairs=proof.final_pairs[:-1])
    )


def test_contractible_simplex_is_not_false_wedge():
    complex_ = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    assert facet_puncture_wedge_certificate(complex_) is None


def test_disconnected_graph_is_not_false_wedge():
    complex_ = clique_complex(nx.empty_graph(4))
    assert facet_puncture_wedge_certificate(complex_) is None


@pytest.mark.parametrize(
    "kw",
    [
        {"max_faces": 0},
        {"max_candidates": 0},
        {"max_removed_facets": 0},
    ],
)
def test_invalid_search_budget(kw):
    complex_ = clique_complex(nx.cycle_graph(4))
    with pytest.raises(ValueError, match="positive"):
        facet_puncture_wedge_certificate(complex_, **kw)
