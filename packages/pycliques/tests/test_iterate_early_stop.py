"""Early stopping at each pared iterate for the standard classifier."""

import networkx as nx
import pycliques.small as small
import pytest


def test_special_octahedron_at_three_does_not_request_four(monkeypatch):
    graphs = [
        nx.cycle_graph(6),
        nx.cycle_graph(7),
        nx.cycle_graph(8),
        nx.complete_multipartite_graph(2, 2, 2, 2),
    ]
    sequences = []

    class GuardedSequence:
        def __init__(self, graph, bound=30):
            assert bound == 120
            self.requested = []
            self.exhausted = False
            sequences.append(self)

        @property
        def graph_count(self):
            return max(self.requested, default=0) + 1

        def __getitem__(self, index):
            self.requested.append(index)
            if index >= len(graphs):
                pytest.fail("Classifier attempted K^4 after an O4 at K^3")
            return graphs[index]

    monkeypatch.setattr(small, "CliqueSequence", GuardedSequence)
    monkeypatch.setattr(small, "completely_pared_graph", lambda graph: graph)
    monkeypatch.setattr(small, "_classify_reference_graph", lambda graph: None)
    monkeypatch.setattr(
        small, "_classify_suspension_2_coaffination", lambda graph: None
    )
    monkeypatch.setattr(small, "is_clique_helly", lambda graph: False)
    monkeypatch.setattr(small, "retracts", lambda graph, target: False)

    result = small.classify_clique_behavior(
        graphs[0], tries=9, bound=120, clockwork_depth=0, retraction_depth=2
    )
    assert result.verdict is small.Verdict.DIVERGENT
    assert result.reason == (
        "eventually has a special octahedron (index 3, dimension 4)"
    )
    assert result.bound_exceeded is False
    assert result.iterations_checked == 4
    assert sequences[0].requested == [0, 1, 2, 3]


def test_helly_certificate_precedes_special_search_and_next_iterate(monkeypatch):
    graphs = [nx.path_graph(7), nx.path_graph(8), nx.path_graph(9)]
    requested = []
    tested_special = []
    monkeypatch.setattr(small, "is_clique_helly", lambda graph: graph is graphs[2])

    def fake_special(graph):
        tested_special.append(graph)
        return None

    monkeypatch.setattr(small, "special_octahedra_dimension", fake_special)

    class Sequence:
        def __getitem__(self, index):
            requested.append(index)
            if index > 2:
                pytest.fail("Advanced beyond an already-Helly iterate")
            return graphs[index]

    result = small._test_iterate_certificates(Sequence(), 9, 0, ())
    assert result == (
        small.Verdict.CONVERGENT,
        "is eventually Helly (index 2)",
        None,
    )
    assert requested == [0, 1, 2]
    assert tested_special == graphs[:2]


def test_first_iterate_retraction_precedes_next_iterate(monkeypatch):
    graphs = [nx.path_graph(7), nx.path_graph(8)]
    target = nx.cycle_graph(5)
    tested_special = []
    requested = []

    monkeypatch.setattr(small, "is_clique_helly", lambda graph: False)
    monkeypatch.setattr(
        small, "special_octahedra_dimension",
        lambda graph: tested_special.append(graph) or None,
    )
    monkeypatch.setattr(
        small, "retracts",
        lambda graph, candidate: graph is graphs[1] and candidate is target,
    )

    class Sequence:
        def __getitem__(self, index):
            requested.append(index)
            if index > 1:
                pytest.fail("Advanced beyond an established retraction")
            return graphs[index]

    result = small._test_iterate_certificates(
        Sequence(), 9, 2, ((target, "retracts to C_5"),)
    )
    assert result == (
        small.Verdict.DIVERGENT,
        "retracts to C_5 (pared iterate 1)",
        None,
    )
    assert requested == [0, 1]
    assert tested_special == [graphs[0]]


def test_early_iterate_certificate_under_small_bound(monkeypatch):
    # A special O4 is certified on the input without attempting to build K(G).
    graph = nx.complete_multipartite_graph(2, 2, 2, 2)
    class NoNextIterate:
        def __getitem__(self, i):
            if i != 0:
                pytest.fail("Advancing was unnecessary after certificate at 0")
            return graph

    monkeypatch.setattr(small, "is_clique_helly", lambda graph: False)
    result = small._test_iterate_certificates(NoNextIterate(), 9, 0, ())
    assert result == (
        small.Verdict.DIVERGENT,
        "eventually has a special octahedron (index 0, dimension 4)",
        None,
    )
