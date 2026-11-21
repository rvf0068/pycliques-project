"""Negative-path coverage for the opt-in snub embedding transport.

These tests are intentionally cheap: the independent full K^4 reconstruction
is already tested in test_small_snub_seeded.py.
"""

import networkx as nx
import pytest
from pycliques.known_octahedral_witnesses import (
    _labeled_f44,
    snub_o24_matches_source,
    snub_o24_pairs_for_iterates,
)
from pycliques.retractions import BoundedRetractionResult, RetractionSearchStatus
from pycliques.small import CliqueSequence, Verdict, _test_iterate_certificates


def _shaped_iterates() -> list[nx.Graph]:
    return [_labeled_f44(), *(nx.empty_graph(n) for n in (12, 20, 56, 1076))]


def test_snub_transport_rejects_correct_order_but_wrong_edge_count() -> None:
    graphs = _shaped_iterates()
    graphs[0] = nx.cycle_graph(8)
    assert snub_o24_pairs_for_iterates(graphs) is None


def test_snub_transport_rejects_nonisomorphic_18_edge_source() -> None:
    graph = nx.complete_bipartite_graph(3, 5)
    graph.add_edges_from(((0, 1), (0, 2), (1, 2)))
    assert graph.number_of_edges() == 18
    assert not snub_o24_matches_source(graph)
    graphs = _shaped_iterates()
    graphs[0] = graph
    assert snub_o24_pairs_for_iterates(graphs) is None


def test_snub_transport_rejects_non_nested_clique_labels() -> None:
    # The first iterate has integers instead of frozensets.
    assert snub_o24_pairs_for_iterates(_shaped_iterates()) is None


def test_snub_transport_rejects_missing_nested_vertex() -> None:
    graphs = _shaped_iterates()
    bad = nx.Graph()
    bad.add_nodes_from(frozenset((99, i)) for i in range(12))
    graphs[1] = bad
    assert snub_o24_pairs_for_iterates(graphs) is None


def test_seeded_rule_handles_missing_iterate() -> None:
    source = _labeled_f44()

    class Missing(CliqueSequence):
        def __getitem__(self, index: int) -> nx.Graph | None:
            return source if index == 0 else None

    assert _test_iterate_certificates(
        Missing(source),
        tries=0,
        retraction_depth=0,
        targets=(),
        octahedron_retract_presets=("snub_o24",),
    ) is None


def test_seeded_rule_handles_missing_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _labeled_f44()
    monkeypatch.setattr("pycliques.small.snub_o24_pairs_for_iterates", lambda _: None)

    class Present(CliqueSequence):
        def __getitem__(self, index: int) -> nx.Graph:
            return source

    assert _test_iterate_certificates(
        Present(source),
        tries=0,
        retraction_depth=0,
        targets=(),
        octahedron_retract_presets=("snub_o24",),
    ) is None


def test_seeded_rule_reports_resource_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _labeled_f44()
    target = nx.octahedral_graph()
    pairs = tuple(nx.non_edges(target))
    monkeypatch.setattr("pycliques.small.snub_o24_pairs_for_iterates", lambda _: pairs)
    monkeypatch.setattr(
        "pycliques.small.find_octahedral_retraction",
        lambda *_args, **_kwargs: BoundedRetractionResult(
            RetractionSearchStatus.LIMIT_REACHED, None, None, 1, 0
        ),
    )

    class Present(CliqueSequence):
        def __getitem__(self, index: int) -> nx.Graph:
            return source if index == 0 else target

    result = _test_iterate_certificates(
        Present(source),
        tries=0,
        retraction_depth=0,
        targets=(),
        octahedron_retract_presets=("snub_o24",),
    )
    assert result is not None
    assert result[0] is Verdict.INDETERMINATE
    assert "resource limits" in result[1]
