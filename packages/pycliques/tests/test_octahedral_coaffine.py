"""Regression checks for exact unseeded coaffine octahedral retraction CSP."""

from __future__ import annotations

import networkx as nx
import pycliques
import pytest
from pycliques.retractions import RetractionSearchStatus


@pytest.fixture(scope="module")
def f45_pared_iterates() -> list[nx.Graph]:
    graph = nx.from_graph6_bytes(b"HCZTf\\}")
    result = [pycliques.completely_pared_graph(graph)]
    for _ in range(4):
        nxt = pycliques.clique_graph(result[-1])
        assert nxt is not None
        result.append(pycliques.completely_pared_graph(nxt))
    assert [(len(g), g.number_of_edges()) for g in result] == [
        (9, 21),
        (14, 59),
        (23, 158),
        (46, 740),
        (202, 16289),
    ]
    return result


def test_f45_fourth_pared_iterate_retracts_to_unseeded_o9(
    f45_pared_iterates: list[nx.Graph],
) -> None:
    host = f45_pared_iterates[4]
    result = pycliques.find_octahedral_retraction_coaffine(
        host, max_states=10_000, time_limit=30
    )
    assert result.status is RetractionSearchStatus.FOUND
    assert result.dimension == 9
    assert result.antipodal_pairs is not None
    assert result.retraction is not None
    assert result.inclusion is not None
    assert result.endomorphism is not None
    assert result.states_explored < 10_000
    assert pycliques.verify_induced_octahedron(host, result.antipodal_pairs)
    target = host.subgraph(v for p in result.antipodal_pairs for v in p)
    assert pycliques.verify_retraction_witness(
        host, target, (result.retraction, result.inclusion)
    )
    assert all(
        not host.has_edge(v, result.endomorphism[v]) and v != result.endomorphism[v]
        for v in host
    )
    assert all(
        result.retraction[v] == result.endomorphism[result.endomorphism[v]]
        for v in host
    )


def test_f45_third_pared_iterate_has_no_octahedral_retract(
    f45_pared_iterates: list[nx.Graph],
) -> None:
    result = pycliques.find_octahedral_retraction_coaffine(
        f45_pared_iterates[3], max_states=5000, time_limit=15
    )
    assert result.status is RetractionSearchStatus.EXHAUSTED
    assert result.retraction is None


def test_octahedron_itself_yields_an_o3_retraction() -> None:
    octahedron = nx.relabel_nodes(
        nx.octahedral_graph(), {i: ("vertex", i) for i in range(6)}
    )
    result = pycliques.find_octahedral_retraction_coaffine(octahedron)
    assert result.status is RetractionSearchStatus.FOUND
    assert result.dimension == 3
    assert result.retraction == {v: v for v in octahedron}
    assert result.endomorphism is not None


def test_no_target_on_complete_or_cycle_graph() -> None:
    for graph in (nx.complete_graph(6), nx.cycle_graph(6)):
        result = pycliques.find_octahedral_retraction_coaffine(
            graph, max_states=10_000, time_limit=10
        )
        assert result.status is RetractionSearchStatus.EXHAUSTED
        assert result.retraction is None


def test_bounded_resource_limit_never_means_no_retraction(
    f45_pared_iterates: list[nx.Graph],
) -> None:
    result = pycliques.find_octahedral_retraction_coaffine(
        f45_pared_iterates[4], max_states=1
    )
    assert result.status is RetractionSearchStatus.LIMIT_REACHED
    assert result.states_explored == 1
    assert result.retraction is None
    assert result.dimension is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_dimension": 2},
        {"min_dimension": True},
        {"max_states": 0},
        {"max_states": True},
        {"time_limit": -1},
        {"time_limit": float("nan")},
        {"time_limit": True},
    ],
)
def test_invalid_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        pycliques.find_octahedral_retraction_coaffine(nx.octahedral_graph(), **kwargs)


def test_input_graph_validation() -> None:
    with pytest.raises(ValueError):
        pycliques.find_octahedral_retraction_coaffine(nx.DiGraph([(0, 1)]))
    with pytest.raises(ValueError):
        pycliques.find_octahedral_retraction_coaffine(nx.MultiGraph([(0, 1)]))
    with pytest.raises(ValueError):
        pycliques.find_octahedral_retraction_coaffine(nx.Graph([(0, 0)]))
