"""Integral Morse plus genuine cup-product obstruction for 2/4-cell complexes."""

import base64
from dataclasses import replace
from importlib import import_module

import networkx as nx
from pycombtop import (
    FlagReductionCertificate,
    Theorem,
    homotopy_type_with_verdict,
    hopf_wedge_certificate,
    reduced_hopf_wedge_certificate,
    symbolic_integral_morse_certificate,
    verify_hopf_wedge,
    verify_reduced_hopf_wedge,
)
from pycombtop.hopf_wedge import (
    _cup_witness,
    _from_morse,
    _hopf_shape,
    _mod2_coboundary_basis,
    _not_coboundary,
    _unit_boundary,
)
from pycombtop.symbolic_integral_morse import SymbolicIntegralMorseCertificate

# Fixed canonical 61-vertex, 417-edge core of K^4(Susp(C10)).
# No census-specific logic is included in pycombtop itself.
C10_CORE_GRAPH6_B64 = (
    "fGBDRz8/fkd4eGNCTm4/WV92ez8/bz8/Pz9HP3s/Pz8/Pz8/Xz9GWz8/eXc/QkY/Pz8/Pz8/"
    "RD8/P0VHPz9BeT8/P0NXPz8/UFc/Pz9JXz8/R29jPz9LfX4/Pz9OXndBb0M/P0NGT08/"
    "QG8/RlZTcz9dbz8/Qj8/QUo/P09CXz8/P1k/P0NMV0NfQXE/Pz9DQHdeQz8/WUM/"
    "P2FyeEBCPz9HWz9fQD9dT0A/QU9lPz9gV0BkUm8/QUVtU0BMP09PPz9BT2M/Pz9HXmtBUkNR"
    "c09fQF9xZT8/eWFmRFJXTmd3Pz9AUF9HQGA/Pz9MQUFHQlE/Pz9ZYz9HPz9ZT2JkQU9f"
    "Qm9fPz9LVT9tPz8/QE97YUJTUEBvPz8/R0NAVEZua08/QD9SdHRPP0FCP0dfSUNP"
)


def c10_core():
    return nx.from_graph6_bytes(base64.b64decode(C10_CORE_GRAPH6_B64))


def _fake_morse(**overrides):
    args = dict(
        critical={0: 1, 1: 1, 2: 1, 3: 2, 4: 8},
        boundaries={
            1: ((-1,),),
            2: ((0,),),
            3: ((0, 0),),
            4: ((1, 0, -1, 0, 0, 0, 0, 0), (0, 0, 0, 0, -1, 0, 0, 1)),
        },
        betti={2: 1, 4: 6},
        torsion={},
        wedge_spheres=None,
        total_faces=3717,
        diagram_nodes=1,
        gradient_states=1,
        critical_masks=(7,),
    )
    args.update(overrides)
    return SymbolicIntegralMorseCertificate(**args)


def test_c10_exact_morse_hopf_cup_square_and_homotopy_classifier():
    graph = c10_core()
    assert (len(graph), graph.number_of_edges()) == (61, 417)
    morse = symbolic_integral_morse_certificate(graph)
    assert morse is not None
    assert morse.critical == {0: 1, 1: 1, 2: 1, 3: 2, 4: 8}
    assert morse.betti == {2: 1, 4: 6}
    assert morse.torsion == {}
    assert morse.boundaries[4] == (
        (1, 0, -1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, -1, 0, 0, 1),
    )
    assert morse.wedge_spheres is None  # boundary ranks alone are inadequate
    proof = hopf_wedge_certificate(graph)
    assert proof is not None
    assert proof.spheres == {2: 1, 4: 6}
    assert proof.critical == morse.critical
    assert proof.triangle_indices == (47, 50, 59)
    assert verify_hopf_wedge(graph, proof)
    verdict = homotopy_type_with_verdict(graph, max_vertices=80, max_simplices=15_000)
    assert verdict.is_exact
    assert verdict.wedge.spheres == {2: 1, 4: 6}
    assert verdict.reason == Theorem.SYMBOLIC_HOPF_CUP_SQUARE


def test_certificate_tampering_and_label_independence():
    graph = c10_core()
    proof = hopf_wedge_certificate(graph)
    assert proof is not None
    assert not verify_hopf_wedge(graph, replace(proof, spheres={2: 1, 4: 5}))
    assert not verify_hopf_wedge(graph, replace(proof, betti={2: 1}))
    assert not verify_hopf_wedge(graph, replace(proof, triangle_indices=(0, 1, 2)))
    assert not verify_hopf_wedge(graph, replace(proof, triangle_count=1))
    changed = graph.copy()
    changed.add_edge(*next(nx.non_edges(changed)))
    assert not verify_hopf_wedge(changed, proof)
    mixed = nx.relabel_nodes(graph, {v: ("node", v) if v % 2 else v for v in graph})
    # Dong matchings depend on vertex order. Relabeling may change
    # which matching is found, but must never produce an invalid verdict.
    renamed = hopf_wedge_certificate(mixed)
    if renamed is not None:
        assert renamed.spheres == proof.spheres
        assert verify_hopf_wedge(mixed, renamed)


def test_proof_rejects_bad_morse_boundary_or_torsion():
    assert _hopf_shape(_fake_morse()) == {2: 1, 4: 6}
    cases = [
        {"critical": {2: 1, 3: 2, 4: 8, 5: 1}},
        {"critical": {0: 2, 1: 1, 2: 1, 3: 2, 4: 8}},
        {"critical": {2: 2, 3: 2, 4: 8}},
        {"critical": {0: 1, 1: 1, 2: 1, 3: 2}},
        {
            "boundaries": {
                1: ((2,),),
                2: ((0,),),
                3: ((0, 0),),
                4: ((1, 0, -1, 0, 0, 0, 0, 0), (0, 0, 0, 0, -1, 0, 0, 1)),
            }
        },
        {
            "boundaries": {
                1: ((-1,),),
                2: ((1,),),
                3: ((0, 0),),
                4: ((1, 0, -1, 0, 0, 0, 0, 0), (0, 0, 0, 0, -1, 0, 0, 1)),
            }
        },
        {
            "boundaries": {
                1: ((-1,),),
                2: ((0,),),
                3: ((1, 0),),
                4: ((1, 0, -1, 0, 0, 0, 0, 0), (0, 0, 0, 0, -1, 0, 0, 1)),
            }
        },
        {
            "boundaries": {
                1: ((-1,),),
                2: ((0,),),
                3: ((0, 0),),
                4: ((2, 0, -2, 0, 0, 0, 0, 0), (0, 0, 0, 0, -1, 0, 0, 1)),
            }
        },
        {"torsion": {3: (2,)}},
        {"betti": {2: 1, 4: 5}},
    ]
    for changes in cases:
        assert _hopf_shape(_fake_morse(**changes)) is None


def test_bad_rectangular_boundary_and_missing_certificate():
    assert not _unit_boundary(((1, 0), (0, 1), (0, 0)), 3, 2)
    critical = {0: 1, 1: 1, 2: 1, 3: 9, 4: 8}
    boundaries = {
        1: ((-1,),),
        2: ((0,),),
        3: (tuple(0 for _ in range(9)),),
        4: tuple((0,) * 8 for _ in range(9)),
    }
    assert _hopf_shape(_fake_morse(critical=critical, boundaries=boundaries)) is None
    assert not verify_hopf_wedge(c10_core(), None)
    assert hopf_wedge_certificate(c10_core(), max_triangles=20_001) is None
    assert hopf_wedge_certificate(c10_core(), max_nodes=250_001) is None


def test_opt_in_reduced_hopf_proof_and_replay(monkeypatch):
    graph = c10_core()
    identity = FlagReductionCertificate((), tuple(graph), tuple(graph.edges))
    hopf_module = import_module("pycombtop.hopf_wedge")
    monkeypatch.setattr(
        hopf_module, "reduce_flag_graph", lambda *args, **kwargs: (graph, identity)
    )
    proof = reduced_hopf_wedge_certificate(graph)
    assert proof is not None
    assert proof.spheres == {2: 1, 4: 6}
    assert verify_reduced_hopf_wedge(graph, proof)
    assert not verify_reduced_hopf_wedge(graph, replace(proof, spheres={2: 1, 4: 7}))
    altered = replace(identity, remaining_vertices=(0,))
    assert not verify_reduced_hopf_wedge(graph, replace(proof, reduction=altered))


def test_cup_witness_rejects_null_class_and_large_inputs():
    sphere = nx.octahedral_graph()
    morse = symbolic_integral_morse_certificate(sphere)
    assert morse is not None
    witness = _cup_witness(sphere, morse, max_triangles=100)
    assert witness is not None  # a maximal triangle is an H^2 generator
    assert witness[1] == 8
    assert _cup_witness(sphere, morse, max_triangles=0) is None
    assert _cup_witness(sphere, morse, max_triangles=1) is None
    tetra = nx.complete_graph(4)
    assert _cup_witness(tetra, morse, max_triangles=10) is None
    contractible = nx.complete_graph(4)
    contractible.add_edges_from(((4, 0), (4, 1)))
    # Maximal triangle (0,1,4) exists but its indicator is a coboundary.
    assert _cup_witness(contractible, morse, max_triangles=20) is None
    basis = _mod2_coboundary_basis([(0, 1), (0, 2), (1, 2)], [(0, 1, 2)])
    assert not _not_coboundary(basis, 1)
    assert _not_coboundary({}, 1)


def test_cup_strategy_fails_closed_when_not_applicable():
    assert hopf_wedge_certificate(nx.Graph()) is None
    assert hopf_wedge_certificate(nx.complete_graph(4)) is None
    assert hopf_wedge_certificate(nx.octahedral_graph()) is None
    assert hopf_wedge_certificate(nx.empty_graph(81)) is None
    assert hopf_wedge_certificate(nx.cycle_graph(5), max_triangles=0) is None
    looped = nx.Graph([(0, 0), (0, 1)])
    assert hopf_wedge_certificate(looped) is None
    assert not verify_hopf_wedge(looped, None)


def test_hopf_boundary_and_topological_rejection_branches():
    """Cover false positives at the integral Morse / cup-product boundary."""
    # A zero-row boundary is vacuously surjective; malformed or
    # nonunit matrices must not certify cellular cancellation.
    assert _unit_boundary(None, 0, 2)
    assert not _unit_boundary(None, 1, 1)
    assert not _unit_boundary(((1,),), 2, 2)
    assert not _unit_boundary(((1, 0),), 1, 1)
    assert not _unit_boundary(((2,),), 1, 1)

    # Genuine 2/4-cell models can have no 0/1 or 3/4 cancellation pairs.
    bare = _fake_morse(
        critical={2: 1, 4: 2},
        boundaries={},
        betti={2: 1, 4: 2},
    )
    assert _hopf_shape(bare) == {2: 1, 4: 2}
    assert _hopf_shape(_fake_morse(critical={})) is None
    assert _hopf_shape(
        _fake_morse(boundaries={1: ((-1,),), 2: ((0,),), 3: ((0, 0),)})
    ) is None

    # The obstruction requires a *connected* complex with a genuine
    # maximal triangular cocycle; no unverified homotopy verdict is returned.
    disconnected = nx.empty_graph(2)
    assert _from_morse(disconnected, _fake_morse(), max_triangles=20) is None
    reference = nx.octahedral_graph()
    morse = symbolic_integral_morse_certificate(reference)
    assert morse is not None
    assert _cup_witness(nx.Graph(), morse, max_triangles=20) is None
    assert _cup_witness(nx.cycle_graph(5), morse, max_triangles=20) is None
    assert _cup_witness(
        reference, morse, max_triangles=20, preferred=((97, 98, 99),)
    ) is not None


def test_hopf_public_limits_and_inconclusive_reduction():
    """No invalid graph or budget is accepted as a proof."""
    assert hopf_wedge_certificate(nx.DiGraph([(0, 1)])) is None
    assert hopf_wedge_certificate(nx.MultiGraph([(0, 1)])) is None
    cycle = nx.cycle_graph(5)
    assert hopf_wedge_certificate(cycle, max_vertices=-1) is None
    assert hopf_wedge_certificate(cycle, max_nodes=0) is None
    assert hopf_wedge_certificate(cycle, max_gradient_states=0) is None
    assert hopf_wedge_certificate(cycle, max_gradient_states=200_001) is None
    assert reduced_hopf_wedge_certificate(nx.Graph()) is None
    assert reduced_hopf_wedge_certificate(nx.empty_graph(1001)) is None
    assert reduced_hopf_wedge_certificate(nx.cycle_graph(5)) is None
