"""Per-root cache correctness, scoping and opt-out regression tests."""

import networkx as nx
import pycliques.small as small
import pytest
from pycliques.small import ReferenceStatus, Verdict


def _count_nested_calls(monkeypatch):
    real = small._classify_clique_behavior_uncached
    calls = []

    def counted(graph, **kwargs):
        calls.append(frozenset(graph))
        return real(graph, **kwargs)

    monkeypatch.setattr(small, "_classify_clique_behavior_uncached", counted)
    real_reference = small._classify_reference_graph

    def extra_repeated_work(graph):
        # A precisely identical inner problem is visited twice. This
        # exercises the public recursive call under one root classification.
        if len(graph) == 5:
            first = small.classify_clique_behavior(nx.cycle_graph(4), tries=1)
            second = small.classify_clique_behavior(nx.cycle_graph(4), tries=1)
            assert first.verdict is second.verdict is Verdict.CONVERGENT
        return real_reference(graph)

    monkeypatch.setattr(small, "_classify_reference_graph", extra_repeated_work)
    return calls


def test_repeated_nested_classifications_use_one_result(monkeypatch):
    calls = _count_nested_calls(monkeypatch)
    stats = {}
    result = small.classify_clique_behavior(
        nx.cycle_graph(5), tries=1, cache_stats=stats
    )
    assert result.verdict is Verdict.CONVERGENT
    assert stats["hits"] == 1
    assert stats["entries"] == 2
    assert len(calls) == 2  # root + single computed inner problem


def test_opt_out_preserves_verdict_without_cache_hits(monkeypatch):
    calls = _count_nested_calls(monkeypatch)
    stats = {}
    result = small.classify_clique_behavior(
        nx.cycle_graph(5), tries=1, recursive_cache=False, cache_stats=stats
    )
    assert result.verdict is Verdict.CONVERGENT
    assert stats == {"hits": 0, "misses": 0, "entries": 0}
    assert len(calls) == 3


def test_cache_is_not_shared_across_separate_roots():
    for _ in range(2):
        stats = {}
        result = small.classify_clique_behavior(nx.cycle_graph(4), cache_stats=stats)
        assert result.verdict is Verdict.CONVERGENT
        assert stats == {"hits": 0, "misses": 1, "entries": 1}


def test_input_labels_are_part_of_cache_key(monkeypatch):
    calls = []
    real = small._classify_clique_behavior_uncached

    def counted(graph, **kwargs):
        calls.append(frozenset(graph))
        return real(graph, **kwargs)

    monkeypatch.setattr(small, "_classify_clique_behavior_uncached", counted)
    real_reference = small._classify_reference_graph

    def evaluate_two_labels(graph):
        if len(graph) == 5:
            a = small.classify_clique_behavior(nx.cycle_graph(4), tries=1)
            relabelled = nx.relabel_nodes(
                nx.cycle_graph(4), {i: f"v{i}" for i in range(4)}
            )
            b = small.classify_clique_behavior(relabelled, tries=1)
            assert a.verdict is b.verdict is Verdict.CONVERGENT
        return real_reference(graph)

    monkeypatch.setattr(small, "_classify_reference_graph", evaluate_two_labels)
    stats = {}
    small.classify_clique_behavior(nx.cycle_graph(5), tries=1, cache_stats=stats)
    assert stats["hits"] == 0
    assert len(calls) == 3


def test_cache_resets_after_exception(monkeypatch):
    original = small._classify_reference_graph

    def fail(_graph):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(small, "_classify_reference_graph", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        small.classify_clique_behavior(nx.cycle_graph(5))
    monkeypatch.setattr(small, "_classify_reference_graph", original)
    stats = {}
    result = small.classify_clique_behavior(nx.cycle_graph(4), cache_stats=stats)
    assert result.verdict is Verdict.CONVERGENT
    assert stats["hits"] == 0


def test_reference_registry_changes_between_root_calls(monkeypatch):
    original = dict(small._REFERENCE_GRAPHS)
    graph = nx.cycle_graph(5)
    try:
        small.clear_reference_graphs()
        small.register_reference_graph(
            graph, status=ReferenceStatus.CONJECTURED, label="conditional"
        )
        conditional = small.classify_clique_behavior(graph, tries=1)
        assert conditional.verdict is Verdict.CONVERGENT
        small.clear_reference_graphs()
        small.register_reference_graph(
            graph, status=ReferenceStatus.PROVEN, label="proven"
        )
        proved = small.classify_clique_behavior(graph, tries=1)
        assert proved.verdict is Verdict.DIVERGENT
    finally:
        small._REFERENCE_GRAPHS.clear()
        small._REFERENCE_GRAPHS.update(original)
