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
