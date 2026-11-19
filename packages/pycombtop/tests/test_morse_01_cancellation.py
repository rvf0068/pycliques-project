"""Certified Morse wedges after cancelling a unit critical 0--1 pair."""

import base64
from collections import Counter

import networkx as nx
import pytest
from pycombtop import (
    Theorem,
    homotopy_type_with_verdict,
    symbolic_integral_morse_certificate,
)
from pycombtop.symbolic_integral_morse import _wedge_after_01_cancellation

# The reduced 70-vertex, 342-edge graph from K^4(Susp(C_11)).
# Fixed graph6: independent of the graph-reduction discovery algorithm.
C11_CORE_G6_B64 = (
    "fj9ARWFDX09GWj9HPz9AP0A/QT9ARz9QdWc/P0E/P0NHQFtjQG8/SD8/P0ZfSENjT3pfRz8/"
    "Rz8/P0NfP0FHPz8/Qj8/Pz9LPz8/P3c/Pz9ARz8/Pz9DPz8/Pz8/Pz8/P11gPz9DPz8/Pz9OfUFn"
    "P0E/Pz9CfFJLdEJDPz8/QE8/Pz8/RWFHPz9EXF9TPz8/P1N0Vz9bRz8/QT8/Pz8/S0c/Pz8/QE9D"
    "Pz9LP0FfQWc/Pz8/P29HQD8/P2A/P0dPPz8/Pz98P08/Pz8/SUA/QT9ZP0Q/P1dBPz8/P08/QT9H"
    "Pz8/Pz9IQD9iWT9HQT9JX09gRz8/Qj9HQFE/Pz8/P0dHSUNTP0g/Pz9EQUNwSD8/QFM/Qkk/Y0A/"
    "T0M/Pz9fRz9AfXc/Pz9PX0M/Pz8/QF9AT19PP0s/Pz8/P0NIQT8/Pz9CXz9gR1k/Pz9NVz8/P0dg"
    "Rz8/Pz8/e0Y/YU9fQm9IXz8/Pz9jaE8/Pz9AWz9XQUNkTz8/P1c/QT8/YD9fP0BwQz9AQz9A"
    "Y1BAPz9ERz9BPz9jP18K"
)


def _core():
    return nx.from_graph6_bytes(base64.b64decode(C11_CORE_G6_B64))


def _explicit_augmented_dong(graph):
    """Independent direct matching and full directed Hasse acyclicity check."""
    indexed = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    faces = {0}
    faces.update(
        sum(1 << i for i in face) for face in nx.enumerate_all_cliques(indexed)
    )
    remaining = faces.copy()
    matched_low = {}
    for pivot in range(len(indexed)):
        bit = 1 << pivot
        removed = set()
        for face in remaining:
            if not (face & bit) and (face | bit) in remaining:
                matched_low[face] = face | bit
                removed.add(face)
                removed.add(face | bit)
        remaining.difference_update(removed)
    directed = nx.DiGraph()
    directed.add_nodes_from(faces)
    for face in faces:
        subfaces = face
        while subfaces:
            bit = subfaces & -subfaces
            subfaces ^= bit
            lower = face ^ bit
            if matched_low.get(lower) == face:
                directed.add_edge(lower, face)
            else:
                directed.add_edge(face, lower)
    assert nx.is_directed_acyclic_graph(directed)
    return len(faces), dict(Counter(face.bit_count() - 1 for face in remaining))


def test_c11_core_exact_consecutive_morse_wedge():
    graph = _core()
    assert (len(graph), graph.number_of_edges()) == (70, 342)
    total, critical = _explicit_augmented_dong(graph)
    assert total == 1215
    assert critical == {0: 1, 1: 1, 2: 3, 3: 4}
    certificate = symbolic_integral_morse_certificate(graph)
    assert certificate is not None
    assert certificate.total_faces == total
    assert certificate.critical == critical
    assert certificate.boundaries[1] in (((1,),), ((-1,),))
    assert certificate.boundaries[2] == ((0, 0, 0),)
    assert certificate.boundaries[3] == ((0, 0, 0, 0),) * 3
    assert certificate.torsion == {}
    assert certificate.betti == {2: 3, 3: 4}
    assert certificate.wedge_spheres == {2: 3, 3: 4}
    verdict = homotopy_type_with_verdict(graph, max_vertices=80, max_simplices=100)
    assert verdict.is_exact
    assert verdict.wedge.spheres == {2: 3, 3: 4}
    assert verdict.reason == Theorem.SYMBOLIC_INTEGRAL_MORSE_01_CANCELLATION


def _proof(
    *,
    critical=None,
    boundaries=None,
    betti=None,
    torsion=None,
):
    return _wedge_after_01_cancellation(
        critical if critical is not None else {0: 1, 1: 1, 2: 3, 3: 4},
        boundaries
        if boundaries is not None
        else {
            1: ((1,),),
            2: ((0, 0, 0),),
            3: ((0, 0, 0, 0),) * 3,
        },
        betti if betti is not None else {2: 3, 3: 4},
        torsion if torsion is not None else {},
    )


def test_unit_smith_cancellation_for_nonzero_upper_boundary():
    # A rank-one primitive differential cancels one 2/3-cell pair.
    assert _proof(
        critical={0: 1, 1: 1, 2: 2, 3: 2},
        boundaries={
            1: ((-1,),),
            2: ((0, 0),),
            3: ((1, 0), (0, 0)),
        },
        betti={2: 1, 3: 1},
    ) == {2: 1, 3: 1}
    # Exactly one positive critical dimension is also a wedge.
    assert _proof(
        critical={0: 1, 1: 1, 5: 2},
        boundaries={1: ((1,),)},
        betti={5: 2},
    ) == {5: 2}
    # The 0/1 pair alone must not be confused with two positive sphere cells.
    assert (
        _proof(
            critical={0: 1, 1: 1},
            boundaries={1: ((1,),)},
            betti={},
        )
        is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"boundaries": {1: ((0,),), 2: ((0, 0, 0),), 3: ((0, 0, 0, 0),) * 3}},
        {"boundaries": {1: ((2,),), 2: ((0, 0, 0),), 3: ((0, 0, 0, 0),) * 3}},
        {"boundaries": {1: ((1,),), 2: ((1, 0, 0),), 3: ((0, 0, 0, 0),) * 3}},
        {"boundaries": {1: ((1,),), 2: ((0, 0, 0),), 3: ((2, 0, 0, 0),) * 3}},
        {"boundaries": {1: ((1,),), 2: ((0, 0, 0),)}},
        {"boundaries": {1: ((1,),), 2: ((0, 0, 0),), 3: ((0, 0),) * 3}},
        {"critical": {0: 2, 1: 1, 2: 3, 3: 4}},
        {"critical": {0: 1, 1: 2, 2: 3, 3: 4}},
        {"critical": {-1: 1, 0: 1, 1: 1, 2: 3, 3: 4}},
        {"critical": {0: 1, 1: 1, 2: 3, 4: 4}},
        {"critical": {0: 1, 1: 1, 2: 3, 3: 4, 5: 1}},
        {"critical": {0: 1, 1: 1, 2: 0, 3: 4}},
        {"betti": {2: 3, 3: 5}},
        {"torsion": {2: (2,)}},
    ],
)
def test_unsupported_morse_shapes_refuse_to_certify(changes):
    assert _proof(**changes) is None
