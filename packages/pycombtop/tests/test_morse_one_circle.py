"""One-loop Morse fundamental-group certificate, including the C12 core."""

import base64
from collections import Counter

import networkx as nx
import pytest
from pycombtop import (
    Theorem,
    homotopy_type_with_verdict,
    symbolic_integral_morse_certificate,
)
from pycombtop.symbolic_integral_morse import _wedge_after_one_circle_cancellation
from sympy import ZZ, Matrix
from sympy.matrices.normalforms import smith_normal_form

# Fixed 48-vertex, 163-edge reduced factor of K^4(Susp(C_12)).
# The underlying clique complex has only 387 augmented faces.
C12_CORE_B64 = (
    "bz8/P0BHZD9HUEFEP2xfRz9DcUBBPz9HP0VnQktnT08/P0tSQE8/Pz9lYz9PTz8/P2Bg"
    "P0BAZ2E/Xz9LP0E/Pz8/T0FEPz9PPz8/P2BPRz9AR0U/Pz9LP0BHRz9fb1BBP0NfP2A/"
    "P0s/P0FJP0dPdEA/P0M/QT9FP0BAT09DPz9LPz8/Z2c/X1NjUz8/QF9YP2hIP0FBQ1s/"
    "Pz8/bz9BP2A/P0c/YE9HQz8/X19BPz8/P0s/P18/RD8/Sz8/Cg=="
)


def c12_core():
    return nx.from_graph6_bytes(base64.b64decode(C12_CORE_B64))


def _explicit_augmented_matching(graph):
    """Check a fully enumerated sequential matching, not the ZDD search."""
    faces = {0}
    faces.update(
        sum(1 << i for i in face) for face in nx.enumerate_all_cliques(graph)
    )
    remaining = faces.copy()
    match_low = {}
    for pivot in sorted(graph):
        bit = 1 << pivot
        removed = set()
        for face in remaining:
            if not (face & bit) and face | bit in remaining:
                match_low[face] = face | bit
                removed.update((face, face | bit))
        remaining.difference_update(removed)

    gradient = nx.DiGraph()
    gradient.add_nodes_from(faces)
    for face in faces:
        bits = face
        while bits:
            bit = bits & -bits
            bits ^= bit
            lower = face ^ bit
            if match_low.get(lower) == face:
                gradient.add_edge(lower, face)
            else:
                gradient.add_edge(face, lower)
    assert nx.is_directed_acyclic_graph(gradient)
    return len(faces), dict(Counter(f.bit_count() - 1 for f in remaining))


def _triangle_closure_proves_simply_connected(graph):
    """Independent sufficient pi1 certificate using only triangle relations."""
    tree = nx.bfs_tree(graph, source=min(graph)).to_undirected()
    assert tree.number_of_edges() == len(graph) - 1
    known = {frozenset(e) for e in tree.edges}
    triangles = [q for q in nx.enumerate_all_cliques(graph) if len(q) == 3]
    while True:
        before = len(known)
        for u, v, w in triangles:
            sides = (frozenset((u, v)), frozenset((u, w)), frozenset((v, w)))
            if sum(e in known for e in sides) >= 2:
                known.update(sides)
        if len(known) == before:
            break
    assert len(triangles) == 153
    return len(known) == graph.number_of_edges()


def _smith_factors(boundary):
    smith = smith_normal_form(Matrix(boundary), domain=ZZ)
    return tuple(
        abs(int(smith[i, i]))
        for i in range(min(smith.shape))
        if smith[i, i]
    )


def test_c12_core_one_circle_morse_classification():
    graph = c12_core()
    assert (len(graph), graph.number_of_edges()) == (48, 163)
    faces, critical = _explicit_augmented_matching(graph)
    assert faces == 387
    assert critical == {0: 5, 1: 6, 2: 17, 3: 1}
    assert _triangle_closure_proves_simply_connected(graph)

    result = symbolic_integral_morse_certificate(graph)
    assert result is not None
    assert result.total_faces == faces
    assert result.critical == critical
    assert {d: _smith_factors(matrix) for d, matrix in result.boundaries.items()} == {
        1: (1, 1, 1, 1, 1),
        2: (1,),
        3: (1,),
    }
    assert result.torsion == {}
    assert result.betti == {2: 15}
    assert result.wedge_spheres == {2: 15}
    verdict = homotopy_type_with_verdict(
        graph, max_vertices=80, max_simplices=100
    )
    assert verdict.is_exact
    assert verdict.wedge.spheres == {2: 15}
    assert verdict.reason == Theorem.SYMBOLIC_INTEGRAL_MORSE_ONE_CIRCLE


BASE_CRITICAL = {0: 2, 1: 3, 2: 3, 3: 2}
BASE_BOUNDARIES = {
    1: ((1, 0, 0), (0, 1, 0)),
    2: ((0, 0, 0), (0, 0, 0), (1, 0, 0)),
    3: ((0, 0), (1, 0), (0, 0)),
}
BASE_BETTI = {2: 1, 3: 1}


def _recognize(
    *, critical=None, boundaries=None, betti=None, torsion=None
):
    return _wedge_after_one_circle_cancellation(
        BASE_CRITICAL if critical is None else critical,
        BASE_BOUNDARIES if boundaries is None else boundaries,
        BASE_BETTI if betti is None else betti,
        {} if torsion is None else torsion,
    )


def test_one_loop_primitive_attaching_maps_have_exact_wedges():
    # Two independent critical 0-cells and three critical edges:
    # the restored base 0-cell yields a tree plus one circle.
    assert _recognize() == {2: 1, 3: 1}

    # A one-vertex 1-skeleton with one circle also qualifies.
    assert _recognize(
        critical={1: 1, 2: 2},
        boundaries={2: ((1, 0),)},
        betti={2: 1},
    ) == {2: 1}

    # A nonzero 3-boundary can cancel *all* surviving 2-spheres.
    assert _recognize(
        critical={1: 1, 2: 2, 3: 2},
        boundaries={2: ((1, 0),), 3: ((0, 0), (1, 0))},
        betti={3: 1},
    ) == {3: 1}


@pytest.mark.parametrize(
    "changes",
    [
        {"critical": {0: 2, 1: 4, 2: 3, 3: 2}},
        {"critical": {0: 2, 1: 3, 3: 2}},
        {"critical": {0: 2, 1: 3, 2: 3, 4: 2}},
        {"critical": {0: 2, 1: 3, 2: 0, 3: 2}},
        {"boundaries": {1: BASE_BOUNDARIES[1], 3: BASE_BOUNDARIES[3]}},
        {"boundaries": {2: BASE_BOUNDARIES[2], 3: BASE_BOUNDARIES[3]}},
        {"boundaries": {**BASE_BOUNDARIES, 4: ((0,),)}},
        {"boundaries": {**BASE_BOUNDARIES, 1: ((1, 0), (0, 1))}},
        {"boundaries": {**BASE_BOUNDARIES, 1: ((1, 0, 0), (0, 0, 0))}},
        {"boundaries": {**BASE_BOUNDARIES, 1: ((2, 0, 0), (0, 1, 0))}},
        {"boundaries": {**BASE_BOUNDARIES, 2: ((0, 0, 0),) * 3}},
        {"boundaries": {**BASE_BOUNDARIES, 2: ((0, 0, 0), (0, 0, 0), (2, 0, 0))}},
        {"boundaries": {**BASE_BOUNDARIES, 3: ((0, 0), (2, 0), (0, 0))}},
        {"boundaries": {**BASE_BOUNDARIES, 3: ((0, 0), (1, 0), (1, 0))}},
        {"boundaries": {**BASE_BOUNDARIES, 2: ((1, 0, 0), (0, 0, 0), (1, 0, 0))}},
        {"betti": {2: 2}},
        {"torsion": {1: (2,)}},
    ],
)
def test_one_loop_rule_rejects_unsupported_or_inconsistent_data(changes):
    assert _recognize(**changes) is None
