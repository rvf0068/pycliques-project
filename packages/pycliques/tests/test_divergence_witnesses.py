"""Independent witness verification and conditional-reference regression tests."""

from dataclasses import replace

import networkx as nx
import pycliques.small as small
import pytest
from pycliques.named import suspension_of_cycle
from pycliques.retractions import (
    find_special_octahedron_witness,
    retracts,
    special_octahedra_dimension,
    verify_retraction_witness,
    verify_special_octahedron_witness,
)
from pycliques.small import (
    ReferenceStatus,
    Verdict,
    _classify_reference_dependency,
    clear_reference_graphs,
    register_reference_graph,
)


@pytest.fixture
def empty_references():
    original = dict(small._REFERENCE_GRAPHS)
    clear_reference_graphs()
    try:
        yield
    finally:
        small._REFERENCE_GRAPHS.clear()
        small._REFERENCE_GRAPHS.update(original)


@pytest.mark.parametrize("d", [3, 4, 5])
def test_octahedral_witness_and_corruption(d):
    graph = nx.complete_multipartite_graph(*([2] * d))
    witness = find_special_octahedron_witness(graph)
    assert witness is not None
    assert witness.dimension == d
    assert verify_special_octahedron_witness(graph, witness)
    assert special_octahedra_dimension(graph) == d
    assert not verify_special_octahedron_witness(
        graph, replace(witness, antipodes=witness.antipodes[:-1])
    )
    assert not verify_special_octahedron_witness(
        graph, replace(witness, antipodes=(witness.clique[0],) + witness.antipodes[1:])
    )


def test_witness_on_arbitrary_labels():
    graph = nx.relabel_nodes(
        nx.complete_multipartite_graph(2, 2, 2),
        {0: "a", 1: "b", 2: ("c", 1), 3: ("c", 2), 4: -1, 5: "z"},
    )
    witness = find_special_octahedron_witness(graph)
    assert witness is not None
    assert verify_special_octahedron_witness(graph, witness)


def test_no_witness_on_chordal_graph():
    assert find_special_octahedron_witness(nx.complete_graph(12)) is None
    assert find_special_octahedron_witness(nx.path_graph(12)) is None


def test_retraction_verifier_accepts_witness_and_rejects_corruption():
    host = nx.path_graph(4)
    target = nx.path_graph(3)
    witness = retracts(host, target)
    assert isinstance(witness, tuple)
    assert verify_retraction_witness(host, target, witness)
    projection, inclusion = witness
    bad_projection = dict(projection)
    bad_projection[inclusion[0]] = 1
    assert not verify_retraction_witness(host, target, (bad_projection, inclusion))


def test_special_certificate_replays_on_saved_iterate():
    graph = nx.octahedral_graph()
    result = small._test_iterate_certificates(
        small.CliqueSequence(graph), tries=1, retraction_depth=0, targets=()
    )
    assert result is not None
    verdict, _, cert = result
    assert verdict is Verdict.DIVERGENT
    assert cert is not None
    assert cert.rule == "special_octahedron"
    assert cert.iterate_index == 0
    assert cert.source is graph
    assert cert.special_octahedron is not None
    assert verify_special_octahedron_witness(graph, cert.special_octahedron)


def test_retraction_certificate_replays_on_saved_iterate(monkeypatch):
    graph = suspension_of_cycle(5)
    monkeypatch.setattr(small, "is_clique_helly", lambda _graph: False)
    result = small._test_iterate_certificates(
        small.CliqueSequence(graph),
        tries=1,
        retraction_depth=1,
        targets=((graph.copy(), "known divergent target"),),
    )
    assert result is not None
    verdict, _, cert = result
    assert verdict is Verdict.DIVERGENT
    assert cert is not None
    assert cert.rule == "retracts"
    assert cert.map is not None
    assert cert.iterate_index == 0
    assert verify_retraction_witness(graph, cert.target, cert.map)


def test_proven_reference_wins_over_earlier_conjecture(empty_references):
    host = nx.path_graph(4)
    register_reference_graph(
        nx.path_graph(2), status=ReferenceStatus.CONJECTURED, label="hint"
    )
    register_reference_graph(
        nx.path_graph(3), status=ReferenceStatus.PROVEN, label="proof"
    )
    result = _classify_reference_dependency(host)
    assert result is not None
    assert result[0] is Verdict.DIVERGENT
    assert result[2] is not None
    assert result[2].target_label == "proof"


def test_conjectural_direct_reference_does_not_mask_helly_proof(empty_references):
    # C5 is clique-Helly and has no dominated vertices; the direct
    # conjectural reference is therefore encountered before the Helly proof.
    graph = nx.cycle_graph(5)
    register_reference_graph(
        graph, status=ReferenceStatus.CONJECTURED, label="unproved"
    )
    result = small.classify_clique_behavior(graph, tries=1)
    assert result.verdict is Verdict.CONVERGENT
    assert result.certificate is None


def test_recursive_nontriangle_rule_propagates_retraction_depth(monkeypatch):
    graph = nx.path_graph(3)
    depths = []

    def substitute(target, **kwargs):
        depths.append(kwargs["retraction_depth"])
        return small.CliqueBehavior(
            Verdict.INDETERMINATE, "unresolved", 1, False, target
        )

    monkeypatch.setattr(small, "classify_clique_behavior", substitute)
    small._classify_non_triangle_edge(graph, retraction_depth=2)
    assert depths and set(depths) == {2}
