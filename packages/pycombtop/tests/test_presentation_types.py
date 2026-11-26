"""Regression tests for certified nonspherical homotopy types."""

from dataclasses import replace
from itertools import combinations, permutations

import networkx as nx
import pytest
from pycombtop import (
    CellularPresentation,
    MooreSpaceType,
    ProductType,
    SimplicialComplex,
    SphereType,
    SurfaceType,
    WedgeType,
    clique_complex,
    complex_type_certificate,
    presentation_type_certificate,
    two_complex_type_certificate,
    verify_complex_type,
    verify_presentation_type,
    verify_two_complex_type,
)


@pytest.mark.parametrize(
    ("generators", "relators", "expected"),
    [
        ((1,), (), {SphereType(1): 1}),
        ((1,), ((1, 1), (), (-1, -1)), {MooreSpaceType(2): 1, SphereType(2): 2}),
        ((1,), ((1,) * 6, (1,) * 9), {MooreSpaceType(3): 1, SphereType(2): 1}),
        ((1,), ((1,) * 2, (1,) * 3), {SphereType(2): 1}),
        ((1, 2), ((1, 2, -1, -2),), {SurfaceType(1): 1}),
        (
            (1, 2),
            ((1, 2, -1, -2), (1, 2, 1, -2, -1, -1)),
            {SurfaceType(1): 1, SphereType(2): 1},
        ),
        ((1, 2), ((1, 2, -1, 2),), {SurfaceType(2, False): 1}),
        (
            (1, 2, 3, 4),
            ((1, 2, -1, -2, 3, 4, -3, -4),),
            {SurfaceType(2): 1},
        ),
        ((1, 2, 3), ((1, 1, 2, 2, 3, 3),), {SurfaceType(3, False): 1}),
        (
            (1, 2, 3, 4),
            ((1, 2, -1, -2), (3, 3), ()),
            {
                SurfaceType(1): 1,
                MooreSpaceType(2): 1,
                SphereType(1): 1,
                SphereType(2): 1,
            },
        ),
        (
            (1, 2),
            ((1, 1), (2, 2)),
            {MooreSpaceType(2): 2},
        ),
        # The paired polygon has two vertices, so identifying them adds S1.
        (
            (1, 2, 3),
            ((1, 2, 3, -1, -2, -3),),
            {SurfaceType(1): 1, SphereType(1): 1},
        ),
    ],
)
def test_recognized_presentations(generators, relators, expected):
    p = CellularPresentation(generators, relators)
    cert = presentation_type_certificate(p)
    assert cert is not None
    assert dict(cert.homotopy_type.summands) == expected
    assert verify_presentation_type(p, cert)


def test_inconclusive_mixed_attachments():
    p = CellularPresentation((1, 2), ((1, 2, -1, -2), (1, 1)))
    assert presentation_type_certificate(p) is None
    # Same abelianized relation as a commutator, but not a torus relator.
    q = CellularPresentation((1, 2), ((1, 2, -1, -2) * 2,))
    assert presentation_type_certificate(q) is None


def test_projective_plane_complex_and_tampering():
    facets = [
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 5),
        (1, 4, 6),
        (1, 5, 6),
        (2, 3, 6),
        (2, 4, 5),
        (2, 5, 6),
        (3, 4, 5),
        (3, 4, 6),
    ]
    sc = SimplicialComplex(range(1, 7), facet_set=facets)
    cert = two_complex_type_certificate(sc)
    assert cert is not None
    assert cert.homotopy_type.moore_spaces == {(2, 1): 1}
    assert verify_two_complex_type(sc, cert)
    assert not verify_two_complex_type(sc, replace(cert, homotopy_type=WedgeType()))


def test_product_expression():
    circle = SphereType(1)
    assert ProductType((circle, circle)).to_latex() == r"S^{1}\times S^{1}"


def test_amoeba_projective_plane_wedge():
    edges = tuple(combinations(range(6), 2))
    bits = {edge: 1 << i for i, edge in enumerate(edges)}
    original = ((1, 5), (2, 5), (3, 4))
    copies = {
        sum(bits[tuple(sorted((p[u], p[v])))] for u, v in original)
        for p in permutations(range(6))
    }
    graph = nx.Graph()
    graph.add_nodes_from(copies)
    for mask in copies:
        for a in bits.values():
            if not mask & a:
                continue
            for b in bits.values():
                if mask & b:
                    continue
                other = mask ^ a ^ b
                if other > mask and other in copies:
                    graph.add_edge(mask, other)
    assert len(graph) == 180 and graph.number_of_edges() == 1440
    sc = clique_complex(graph)
    assert sc.dimension() == 7
    cert = complex_type_certificate(sc)
    assert cert is not None
    assert cert.homotopy_type.moore_spaces == {(2, 1): 1}
    assert cert.homotopy_type.spheres == {2: 134}
    assert verify_complex_type(sc, cert)
    assert not verify_complex_type(sc, replace(cert, collapse_pairs=()))
    assert complex_type_certificate(sc, max_bad_faces=0) is None
