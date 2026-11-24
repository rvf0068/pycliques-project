"""Regression tests for generated divergent clockwork target libraries."""

import itertools

import networkx as nx
import pytest
from pycliques.clockwork import (
    ClockworkDivergenceResult,
    ClockworkDivergenceStatus,
    clockwork_divergence_result,
)
from pycliques.clockwork_library import generate_divergent_clockwork_library
from pycliques.retractions import retracts, verify_retraction_witness


def test_historical_library():
    library = generate_divergent_clockwork_library()
    assert library.templates_checked == 52
    assert len(library.targets) == 13
    graphs = [target.graph for target in library.targets]
    assert {len(graph) for graph in graphs} == {9, 10}
    for graph in graphs:
        result = clockwork_divergence_result(graph, fallback_enabled=False)
        assert result.status is ClockworkDivergenceStatus.DIVERGENT
    for graph, other in itertools.combinations(graphs, 2):
        assert not nx.is_isomorphic(graph, other)


def test_target_reconstruction_and_retraction():
    target = generate_divergent_clockwork_library([(1, 1, 1)]).targets[0]
    graph = target.graph
    host = graph.copy()
    vertex = next(iter(graph))
    host.add_edge(vertex, max(graph) + 1)
    witness = retracts(host, graph)
    assert isinstance(witness, tuple)
    assert verify_retraction_witness(host, graph, witness)
    graph.clear()
    assert len(target.graph) == 9


def test_duplicate_size_tuples_and_empty_family():
    library = generate_divergent_clockwork_library([(1, 1, 1)] * 2)
    assert library.templates_checked == 16
    empty = generate_divergent_clockwork_library([])
    assert empty.targets == ()
    assert empty.templates_checked == empty.unresolved_templates == 0


@pytest.mark.parametrize(
    "sizes,crown,limit",
    [
        ([(1, 1)], 2, 100),
        ([(1, 0, 1)], 2, 100),
        ([(1, True, 1)], 2, 100),
        ([(1, 1, 1)], 1, 100),
        ([(1, 1, 1)], True, 100),
        ([(1, 1, 1)], 2, 0),
        ([(1, 1, 1)], 2, 15),
    ],
)
def test_invalid_or_oversized_family(sizes, crown, limit):
    with pytest.raises(ValueError):
        generate_divergent_clockwork_library(
            sizes, crown_size=crown, max_templates=limit
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fallback_max_order": -1},
        {"fallback_max_order": 1.5},
        {"fallback_max_order": True},
        {"fallback_max_order": "9"},
        {"fallback_max_order": None},
        {"fallback_max_candidates": 0},
        {"fallback_max_candidates": 1.5},
        {"fallback_max_candidates": True},
        {"fallback_max_candidates": "10"},
        {"fallback_max_candidates": None},
    ],
)
def test_invalid_recognition_limits(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        generate_divergent_clockwork_library(**kwargs)


def test_other_crown_sizes_and_unresolved_recognition(monkeypatch):
    def unresolved(graph, **kwargs):
        assert len(graph) == 12
        assert kwargs["fallback_enabled"]
        return ClockworkDivergenceResult(
            ClockworkDivergenceStatus.LIMIT_REACHED, None, "test limit"
        )

    monkeypatch.setattr(
        "pycliques.clockwork_library.clockwork_divergence_result", unresolved
    )
    library = generate_divergent_clockwork_library(
        [(1, 1, 1)], crown_size=3, max_templates=48, fallback_enabled=True
    )
    assert library.templates_checked == library.unresolved_templates == 48
    assert library.targets == ()
