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


@pytest.mark.parametrize(
    ("factory", "args"),
    [
        (SphereType, (-1,)),
        (SphereType, (True,)),
        (MooreSpaceType, (1,)),
        (MooreSpaceType, (2, 0)),
        (SurfaceType, (0, False)),
        (SurfaceType, (1, 1)),
        (WedgeType, (((SphereType(1), 0),),)),
    ],
)
def test_invalid_homotopy_expressions(factory, args):
    with pytest.raises(ValueError):
        factory(*args)


def test_expression_rendering_and_multiplicity():
    circle = SphereType(1)
    torus = ProductType((circle, circle))
    wedge = WedgeType(((circle, 2), (circle, 1), (MooreSpaceType(3), 2)))
    assert wedge.spheres == {1: 3}
    assert wedge.moore_spaces == {(3, 1): 2}
    assert SurfaceType(0).to_latex() == "S^{2}"
    assert SurfaceType(1, False).to_latex() == r"\mathbb{RP}^{2}"
    assert MooreSpaceType(3, 2).to_latex() == r"M(\mathbb{Z}/3,2)"
    assert WedgeType().to_latex() == ProductType(()).to_latex() == r"\mathrm{pt}"
    assert ProductType((wedge, circle)).to_latex() == (
        r"\left(\bigvee^{2}S^{1}\vee S^{1}\vee "
        r"\bigvee^{2}M(\mathbb{Z}/3,1)\right)\times S^{1}"
    )
    assert WedgeType(((torus, 1),)).to_latex() == r"\left(S^{1}\times S^{1}\right)"
    repeated = WedgeType(((MooreSpaceType(2), 2), (MooreSpaceType(2), 3)))
    assert repeated.moore_spaces == {(2, 1): 5}


def test_cyclic_conjugation_and_polygon_with_two_vertices():
    p = CellularPresentation((1, 2, 3), ((3, 1, 2, -1, -2, -3),))
    cert = presentation_type_certificate(p)
    assert cert is not None
    assert dict(cert.homotopy_type.summands) == {SurfaceType(1): 1, SphereType(1): 1}
    assert verify_presentation_type(p, cert)
    q = CellularPresentation((1, 2), ((1, 2, 1, 2),))
    cert = presentation_type_certificate(q)
    assert cert is not None
    assert dict(cert.homotopy_type.summands) == {
        MooreSpaceType(2): 1,
        SphereType(1): 1,
    }


def test_additional_relators_on_nonabelian_or_multi_vertex_surface():
    klein = (1, 2, -1, 2)
    assert (
        presentation_type_certificate(CellularPresentation((1, 2), (klein, klein)))
        is None
    )
    word = (1, 2, 3, -1, -2, -3)
    assert (
        presentation_type_certificate(CellularPresentation((1, 2, 3), (word, word)))
        is None
    )


def test_presentation_limits_and_invalid_certificates():
    p = CellularPresentation((1,), ((1, 1),))
    assert presentation_type_certificate(p, max_total_letters=1) is None
    cert = presentation_type_certificate(p)
    assert cert is not None
    assert cert.verdict == r"\(\mathbb{RP}^{2}\)"
    assert not verify_presentation_type(p, None)
    assert not verify_presentation_type(p, replace(cert, homotopy_type=None))
    disconnected = SimplicialComplex(range(2), facet_set=[(0,), (1,)])
    assert not verify_two_complex_type(disconnected, cert)


def test_complex_limits_and_failed_relative_collapse():
    triangle = SimplicialComplex(range(3), facet_set=[range(3)])
    cert = complex_type_certificate(triangle)
    assert cert is not None
    assert cert.verdict == r"\(\mathrm{pt}\)"
    assert verify_complex_type(triangle, cert)
    assert not verify_complex_type(triangle, None)
    with pytest.raises(ValueError):
        complex_type_certificate(triangle, max_target_facets=0)
    with pytest.raises(ValueError):
        complex_type_certificate(triangle, max_bad_faces=-1)
    square = clique_complex(nx.cycle_graph(4))
    assert complex_type_certificate(square, max_target_facets=1) is None
    tetrahedron = SimplicialComplex(range(4), facet_set=[range(4)])
    assert complex_type_certificate(tetrahedron, max_target_facets=1) is None
    sphere = SimplicialComplex(range(5), facet_set=combinations(range(5), 4))
    assert complex_type_certificate(sphere) is None
    assert complex_type_certificate(triangle, max_total_letters=1) is not None
