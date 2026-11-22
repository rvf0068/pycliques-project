"""Two-shore octahedral solver: complete exactness, budgets and witness checks."""

from __future__ import annotations

import random

import networkx as nx
import pycliques
import pytest

PAIRS = [(0, 1), (2, 3), (4, 5)]
FOUND = pycliques.RetractionSearchStatus.FOUND
EXHAUSTED = pycliques.RetractionSearchStatus.EXHAUSTED
LIMIT = pycliques.RetractionSearchStatus.LIMIT_REACHED


def _forced_chain() -> nx.Graph:
    """An outside b0 assignment forces a second outside b1 assignment."""
    graph = pycliques.octahedron(3)
    graph.add_nodes_from(("x", "y"))
    graph.add_edges_from(("x", v) for v in (1, 2, 3, 4, 5))
    graph.add_edges_from(("y", v) for v in (0, 3, 4, 5))
    graph.add_edge("x", "y")
    return graph


def test_two_shore_forced_cascade_constructs_full_retraction() -> None:
    host = _forced_chain()
    result = pycliques.find_octahedral_retraction_shores(
        host, PAIRS, orientation=0, max_states=3
    )
    assert result.status is FOUND
    assert result.states_explored == 3
    assert result.assignments_tried == 2
    assert result.retraction is not None
    assert result.retraction["x"] == 1
    assert result.retraction["y"] == 3
    assert result.inclusion == {v: v for v in range(6)}
    assert pycliques.verify_retraction_witness(
        host, host.subgraph(range(6)), (result.retraction, result.inclusion)
    )


@pytest.mark.parametrize("budget", [1, 2])
def test_two_shore_resource_bounds_are_not_negative(budget: int) -> None:
    result = pycliques.find_octahedral_retraction_shores(
        _forced_chain(), PAIRS, orientation=0, max_states=budget
    )
    assert result.status is LIMIT
    assert result.states_explored == budget
    assert result.assignments_tried == budget - 1
    assert result.retraction is None


def test_two_shore_target_itself_needs_only_initial_state() -> None:
    host = pycliques.octahedron(3)
    for orient in (0, 7, None):
        result = pycliques.find_octahedral_retraction_shores(
            host, PAIRS, orientation=orient, max_states=1
        )
        assert result.status is FOUND
        assert result.states_explored == 1
        assert result.assignments_tried == 0
        assert result.retraction == {v: v for v in host}


def test_two_shore_universal_vertex_is_exact_negative() -> None:
    host = pycliques.octahedron(3)
    host.add_node("universal")
    host.add_edges_from(("universal", v) for v in range(6))
    for orient in (0, 5, None):
        result = pycliques.find_octahedral_retraction_shores(
            host, PAIRS, orientation=orient
        )
        assert result.status is EXHAUSTED
        assert result.states_explored == 1
        assert result.assignments_tried == 0


def test_two_shore_forced_opposite_edge_is_exact_negative() -> None:
    host = pycliques.octahedron(3)
    host.add_nodes_from(("left", "right"))
    host.add_edges_from(("left", v) for v in (0, 2, 3, 4, 5))
    host.add_edges_from(("right", v) for v in (1, 2, 3, 4, 5))
    host.add_edge("left", "right")
    for orient in (0, 5, None):
        result = pycliques.find_octahedral_retraction_shores(
            host, PAIRS, orientation=orient
        )
        assert result.status is EXHAUSTED
        assert result.retraction is None


def test_two_shore_matches_independent_mrv_exact_status_for_random_hosts() -> None:
    """Compare two unrelated exact algorithms for arbitrary target attachments."""
    rng = random.Random(20261122)
    for outer_order in (1, 2, 3):
        for _ in range(50):
            host = pycliques.octahedron(3)
            host.add_nodes_from(range(6, 6 + outer_order))
            for a in range(6, 6 + outer_order):
                for b in range(a):
                    if rng.randrange(2):
                        host.add_edge(a, b)
            reference = pycliques.find_octahedral_retraction(
                host, PAIRS, max_states=10_000, seed=7
            )
            assert reference.status is not LIMIT
            for orientation in (0, 3, 7, None):
                result = pycliques.find_octahedral_retraction_shores(
                    host, PAIRS, orientation=orientation, max_states=10_000
                )
                assert result.status is reference.status, (
                    nx.to_graph6_bytes(host, header=False),
                    orientation,
                    reference.status,
                    result.status,
                )
                assert result.status is not LIMIT
                if result.status is FOUND:
                    assert result.retraction is not None
                    assert result.inclusion is not None
                    assert pycliques.verify_retraction_witness(
                        host,
                        host.subgraph(range(6)),
                        (result.retraction, result.inclusion),
                    )


@pytest.mark.parametrize(
    "options",
    [
        {"max_states": 0},
        {"max_states": True},
        {"max_orientation_samples": 0},
        {"max_orientation_samples": 2.5},
        {"orientation": -1},
        {"orientation": 8},
        {"orientation": True},
        {"orientation": 1.5},
    ],
)
def test_two_shore_invalid_configurations(options: dict) -> None:
    with pytest.raises(ValueError):
        pycliques.find_octahedral_retraction_shores(
            pycliques.octahedron(3), PAIRS, **options
        )


def test_two_shore_invalid_target_reuses_exact_induced_validation() -> None:
    graph = pycliques.octahedron(3)
    with pytest.raises(ValueError):
        pycliques.find_octahedral_retraction_shores(graph, [(0, 1), (2, 3), (4, 8)])
    graph.remove_edge(0, 2)
    with pytest.raises(ValueError):
        pycliques.find_octahedral_retraction_shores(graph, PAIRS)


def test_two_shore_independent_positive_witness_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pycliques.octahedral_shore_retractions.verify_retraction_witness",
        lambda *_: False,
    )
    with pytest.raises(AssertionError, match="invalid witness"):
        pycliques.find_octahedral_retraction_shores(pycliques.octahedron(3), PAIRS)
