"""Symbolic squares, independently known nonzero controls and stable wedges."""

import base64
from collections import defaultdict
from dataclasses import replace
from itertools import combinations

import networkx as nx
import pytest
from pycombtop import (
    SphereType,
    SuspendedHopfType,
    SymbolicSq2Certificate,
    WedgeType,
    symbolic_integral_morse_certificate,
    verify_connected_link_order,
    verify_symbolic_sq2,
)
from pycombtop._symbolic_morse_gradient import matching_pivot, morse_boundaries
from pycombtop._symbolic_steenrod import compute_sq2
from pycombtop.symbolic_dong import _ZDD
from pycombtop.symbolic_integral_morse import _sq2_type

CORE = (
    "c35+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fl9efmNmfnR2fnt7fn5oXn5+"
    "eF5+fmN+fn5ofn5+SH5+fnJ2fkB+fn57bk5+fnltXn5+eGh+fn57eF5+fn5b"
    "Zn5+fntYfn5+fnRMfn5+fkB+fkB+fkpyfkpyfml4fml4fnhofnhofn5NVn5NVn58"
    "cV58cV5+e1h+e1h+fnxSXnxSXn5+fV9OfV9Ofn5nY35nY35+cWpucWpufn5RXV5"
    "RXV5+fVhKfVhKfn5wfnFwfnF+fnl8cVl8cV5+fVxxXVxxXn5+Cg=="
)
ORDER = (
    2,
    5,
    11,
    12,
    13,
    15,
    16,
    17,
    20,
    23,
    29,
    30,
    31,
    33,
    34,
    35,
    3,
    4,
    6,
    7,
    8,
    9,
    10,
    14,
    21,
    22,
    24,
    25,
    26,
    27,
    28,
    32,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    0,
    1,
    18,
    19,
    46,
    49,
    47,
    48,
    50,
    51,
    44,
    45,
)


def test_large_cubic_core_square_and_replay():
    graph = nx.from_graph6_bytes(base64.b64decode(CORE))
    assert (len(graph), graph.number_of_edges()) == (52, 1054)
    options = dict(vertex_order=ORDER, max_critical=128)
    proof = symbolic_integral_morse_certificate(
        graph, **options, max_sq2_states=2_000_000
    )
    assert proof is not None
    assert proof.total_faces == 26_806_756  # includes the empty face
    assert proof.betti == {5: 17, 6: 2, 7: 49}
    assert proof.torsion == {}
    assert proof.sq2 is not None
    assert (proof.sq2.source_rank, proof.sq2.target_rank, proof.sq2.rank) == (17, 49, 0)
    assert proof.sq2.rows == (0,) * 49
    assert proof.sq2.lifted_faces == 6724
    assert proof.wedge_spheres == proof.betti
    assert proof.homotopy_type == WedgeType(
        ((SphereType(5), 17), (SphereType(6), 2), (SphereType(7), 49))
    )
    assert proof.connectivity_order is not None
    assert verify_connected_link_order(graph, proof.connectivity_order)
    assert verify_symbolic_sq2(graph, proof)
    forged = replace(proof, sq2=replace(proof.sq2, rank=1))
    assert not verify_symbolic_sq2(graph, forged)
    limited = symbolic_integral_morse_certificate(graph, **options, max_sq2_states=1)
    assert limited is not None and limited.sq2_limited
    assert limited.sq2 is None
    assert limited.betti == proof.betti
    assert limited.wedge_spheres is None
    assert limited.homotopy_type is None


def _product_facets():
    # A six-vertex triangulation of RP^2. Staircase triangulate its
    # Cartesian square; the two factor degree-one generators are x,y.
    # Sq^2(x*y) = x^2*y^2 != 0, whereas Sq^2(x^2)=Sq^2(y^2)=0.
    rp = [
        tuple(int(c) - 1 for c in s)
        for s in ("123", "124", "135", "146", "156", "236", "245", "256", "345", "346")
    ]
    facets = []
    for a in rp:
        for b in rp:
            for moves in combinations(range(4), 2):
                i = j = 0
                path = [6 * a[i] + b[j]]
                for t in range(4):
                    if t in moves:
                        i += 1
                    else:
                        j += 1
                    path.append(6 * a[i] + b[j])
                facets.append(tuple(path))
    return facets


@pytest.mark.parametrize("suspend", [False, True])
def test_nonzero_square_on_projective_product(suspend):
    facets = _product_facets()
    if suspend:
        facets = [q + (v,) for q in facets for v in (36, 37)]
    all_faces = {0}
    for facet in facets:
        mask = sum(1 << v for v in facet)
        face = mask
        while face:
            all_faces.add(face)
            face = (face - 1) & mask
    n = 38 if suspend else 36
    diagram = _ZDD(n, 250_000)

    def build(family):
        if not family:
            return 0
        union = 0
        for mask in family:
            union |= mask
        if not union:
            return 1
        bit = union & -union
        return diagram.node(
            bit.bit_length() - 1,
            build({m for m in family if not m & bit}),
            build({m ^ bit for m in family if m & bit}),
        )

    try:
        roots = [build(all_faces)]
        for v in range(n):
            roots.append(diagram.pivot(roots[-1], v))
        faces = defaultdict(list)
        for mask in sorted(diagram.masks(roots[-1])):
            faces[mask.bit_count() - 1].append(mask)
        pivot = matching_pivot(diagram, roots)
        boundaries, _, _, _ = morse_boundaries(
            diagram, roots, faces, 200_000, pivot_lookup=pivot
        )
        proof = compute_sq2(faces, boundaries, pivot, 3 if suspend else 2, 2_000_000)
        assert (proof.source_rank, proof.target_rank, proof.rank) == (3, 1, 1)
    finally:
        diagram.pivot.cache_clear()
        diagram.difference.cache_clear()
        diagram.contains_empty.cache_clear()
        diagram.remove_empty.cache_clear()


def test_structured_suspended_hopf_type_and_validation():
    square = SymbolicSq2Certificate(3, 2, 2, (1, 0), 1, 1, 1)
    assert _sq2_type({3: 2, 4: 1, 5: 2}, square) == WedgeType(
        (
            (SphereType(3), 1),
            (SphereType(4), 1),
            (SphereType(5), 1),
            (SuspendedHopfType(3), 1),
        )
    )
    assert SuspendedHopfType(3).to_latex() == r"\Sigma^{1}\mathbb{CP}^2"
    with pytest.raises(ValueError):
        SuspendedHopfType(2)
    with pytest.raises(ValueError):
        symbolic_integral_morse_certificate(nx.path_graph(3), steenrod_degree=2)
    with pytest.raises(ValueError):
        symbolic_integral_morse_certificate(nx.path_graph(3), max_sq2_states=-1)


def test_explicit_square_outside_recognition_range():
    graph = nx.octahedral_graph()
    proof = symbolic_integral_morse_certificate(
        graph, steenrod_degree=2, max_sq2_states=100
    )
    assert proof is not None and proof.sq2 is not None
    assert (proof.sq2.source_rank, proof.sq2.target_rank, proof.sq2.rank) == (1, 0, 0)
    assert proof.sq2_homotopy_type is None
    assert verify_symbolic_sq2(graph, proof)
