"""Regressions for verified facet-greedy discrete Morse wedge certificates."""

import networkx as nx
import pytest

from pycliques import clique_graph
from pycombtop.homotopy_type import Theorem, homotopy_type_with_verdict
from pycombtop.morse_certificates import facet_morse_certificate, graph_morse_certificate

# The original order-9 McKay indices are retained to identify the census rows.
SIX_TIMEOUTS = (
    (108411, "H?o~f~}", {2: 1, 3: 2}, True),
    (145941, "H?~vf~}", {3: 30}, False),
    (239843, "HCrbf~}", {2: 1, 3: 2}, True),
    (256373, "HEzvf^m", {3: 17}, False),
    (256477, "HFzfvz{", {3: 24}, False),
    (256483, "HFzf~z{", {3: 46}, False),
)


@pytest.mark.parametrize("index,g6,expected,hurewicz", SIX_TIMEOUTS)
def test_order9_clique_graph_morse_certificates(index, g6, expected, hurewicz):
    g = nx.from_graph6_bytes(g6.encode("ascii"))
    k = clique_graph(g)
    assert k is not None
    certificate = graph_morse_certificate(k)
    assert certificate is not None, f"order-9 index {index}"
    assert certificate.spheres == expected
    assert certificate.critical[0] == 1
    assert certificate.uses_hurewicz is hurewicz
    assert certificate.matching_pairs > 0


def test_classifier_uses_early_facet_morse_certificate():
    # This clique graph has high dimension and at least 20 facets.
    g = nx.from_graph6_bytes(b"H?~vf~}")
    k = clique_graph(g)
    verdict = homotopy_type_with_verdict(
        k, max_vertices=64, max_simplices=25000
    )
    assert verdict.is_exact and not verdict.is_limited
    assert verdict.wedge.spheres == {3: 30}
    assert verdict.reason == Theorem.FACET_MORSE


def test_clique_graph_with_mixed_critical_dimensions_uses_hurewicz():
    g = nx.from_graph6_bytes(b"H?o~f~}")
    k = clique_graph(g)
    verdict = homotopy_type_with_verdict(
        k, max_vertices=64, max_simplices=25000
    )
    assert verdict.is_exact and not verdict.is_limited
    assert verdict.wedge.spheres == {2: 1, 3: 2}
    assert verdict.reason == Theorem.FACET_MORSE_HUREWICZ


def test_small_known_types_and_bad_budget():
    assert graph_morse_certificate(nx.path_graph(3)).spheres == {}
    assert graph_morse_certificate(nx.cycle_graph(5)).spheres == {1: 1}
    assert graph_morse_certificate(nx.octahedral_graph()).spheres == {2: 1}
    assert graph_morse_certificate(nx.complete_graph(6), max_faces=10) is None
    with pytest.raises(ValueError):
        facet_morse_certificate([{0}], max_faces=0)


def test_invalid_matching_is_never_certified(monkeypatch):
    import pycombtop.morse_certificates as morse

    monkeypatch.setattr(morse, "_valid_acyclic_matching", lambda *_: False)
    assert morse.graph_morse_certificate(nx.cycle_graph(5)) is None


def test_adjacent_dimension_certificate_requires_top_homology(monkeypatch):
    import pycombtop.morse_certificates as morse

    monkeypatch.setattr(morse, "_top_betti_matches", lambda *_args, **_kw: False)
    g = nx.from_graph6_bytes(b"H?o~f~}")
    assert morse.graph_morse_certificate(clique_graph(g)) is None
