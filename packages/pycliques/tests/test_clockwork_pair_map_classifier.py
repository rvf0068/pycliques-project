"""Tests for the theorem_2_6_clockwork_pair_map classifier rule."""

import networkx as nx
import pycliques.small as small
from pycliques.cliques import clique_graph
from pycliques.coaffinations import CoaffinePair
from pycliques.small import (
    Verdict,
    _classify_clockwork_pair_map,
    classify_clique_behavior,
    classify_clockwork_pair_map,
)

# ---------------------------------------------------------------------------
# Test 2 -- target has multiple coaffinations; a failing one must not stop
# the search from finding a working one.
# ---------------------------------------------------------------------------


def test_classify_clockwork_pair_map_tries_multiple_target_coaffinations(
    monkeypatch,
):
    cube = nx.convert_node_labels_to_integers(nx.hypercube_graph(3))
    failing_tau = {7: 0, 6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7}
    working_tau = {3: 0, 2: 1, 1: 2, 0: 3, 7: 4, 6: 5, 5: 6, 4: 7}
    source_pair = CoaffinePair(nx.cycle_graph(4), {0: 2, 1: 3, 2: 0, 3: 1})

    def fake_candidates(graph, radius):
        yield failing_tau
        yield working_tau

    monkeypatch.setattr(small, "candidate_target_coaffinations", fake_candidates)
    monkeypatch.setattr(small, "clockwork_coaffine_pair", lambda m, n: source_pair)

    result = _classify_clockwork_pair_map(cube, max_m=1, max_n=0)

    assert result is not None
    verdict, _, certificate = result
    assert verdict is Verdict.DIVERGENT
    assert certificate is not None
    assert certificate.target_coaffination == working_tau


# ---------------------------------------------------------------------------
# Test 7 -- multiple m values; certificate must report the actual m used.
# ---------------------------------------------------------------------------


def test_classify_clockwork_pair_map_reports_actual_m(monkeypatch):
    def fake_candidates(graph, radius):
        # Only the radius that corresponds to m == 2 has a candidate.
        if radius == 3:
            yield {"y": "y"}

    def fake_source(m, n):
        return CoaffinePair(nx.Graph([("x", "x2")]), {"x": "x", "x2": "x2"})

    def fake_morphism(source_pair, target_pair):
        return {"x": "y", "x2": "y"}

    monkeypatch.setattr(small, "candidate_target_coaffinations", fake_candidates)
    monkeypatch.setattr(small, "clockwork_coaffine_pair", fake_source)
    monkeypatch.setattr(small, "find_pair_morphism", fake_morphism)

    result = _classify_clockwork_pair_map(nx.Graph(), max_m=3, max_n=0)

    assert result is not None
    _, _, certificate = result
    assert certificate is not None
    assert certificate.m == 2
    assert certificate.radius == 3


# ---------------------------------------------------------------------------
# Test 8 -- multiple n values; certificate must report the actual n used.
# ---------------------------------------------------------------------------


def test_classify_clockwork_pair_map_reports_actual_n(monkeypatch):
    def fake_candidates(graph, radius):
        yield {"y": "y"}

    # Distinguish source pairs built for different n by object identity,
    # since the fake source graph is otherwise the same for every n.
    built_pairs = {}

    def fake_source_pair(m, n):
        pair = CoaffinePair(nx.Graph([("x", "x2")]), {"x": "x", "x2": "x2"})
        built_pairs[id(pair)] = n
        return pair

    def fake_morphism(source_pair, target_pair):
        if built_pairs[id(source_pair)] == 2:
            return {"x": "y", "x2": "y"}
        return None

    monkeypatch.setattr(small, "candidate_target_coaffinations", fake_candidates)
    monkeypatch.setattr(small, "clockwork_coaffine_pair", fake_source_pair)
    monkeypatch.setattr(small, "find_pair_morphism", fake_morphism)

    result = _classify_clockwork_pair_map(nx.Graph(), max_m=1, max_n=2)

    assert result is not None
    _, _, certificate = result
    assert certificate is not None
    assert certificate.n == 2


# ---------------------------------------------------------------------------
# Test 9 -- no certificate found; must return None, not CONVERGENT.
# ---------------------------------------------------------------------------


def test_classify_clockwork_pair_map_returns_none_when_no_certificate():
    petersen = nx.petersen_graph()

    result = _classify_clockwork_pair_map(petersen, max_m=3, max_n=3)

    assert result is None
    assert classify_clockwork_pair_map(petersen) is None


# ---------------------------------------------------------------------------
# Test 10 -- classifier integration: K(icosahedral graph) is decided by this
# rule alone (verified: every earlier rule returns None on this graph).
# ---------------------------------------------------------------------------


def test_classify_clique_behavior_uses_clockwork_pair_map_certificate():
    kg = clique_graph(nx.icosahedral_graph())

    result = classify_clique_behavior(kg)

    assert result.verdict is Verdict.DIVERGENT
    assert result.certificate is not None
    assert result.certificate.rule == "theorem_2_6_clockwork_pair_map"
    assert result.certificate.m == 2
    assert result.certificate.n == 0
    assert result.certificate.radius == 3


# ---------------------------------------------------------------------------
# Test 11 -- fall-through: this rule failing does not prevent other rules
# (here, the eventually-Helly test) from classifying the graph.
# ---------------------------------------------------------------------------


def test_clockwork_pair_map_failure_falls_through_to_other_rules():
    petersen = nx.petersen_graph()

    assert _classify_clockwork_pair_map(petersen) is None

    result = classify_clique_behavior(petersen)

    assert result.verdict is Verdict.CONVERGENT
