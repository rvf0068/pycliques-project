"""Regression and brute-force checks for general coaffine and stiff searches."""

from __future__ import annotations

from itertools import combinations, product

import networkx as nx
import pytest
from pycliques import (
    RetractionSearchStatus,
    clique_graph,
    completely_pared_graph,
    find_coaffine_endomorphism,
    search_proper_stiff_retraction,
    verify_coaffine_endomorphism,
    verify_retraction_witness,
)


def test_general_coaffine_matches_brute_force() -> None:
    for graph in nx.graph_atlas_g():
        if len(graph) > 5:
            break
        nodes = list(graph)
        domains = [
            [w for w in nodes if w != v and not graph.has_edge(v, w)]
            for v in nodes
        ]
        expected = any(
            all(
                images[u] == images[v] or graph.has_edge(images[u], images[v])
                for u, v in graph.edges()
            )
            for images in product(*domains)
        )
        result = find_coaffine_endomorphism(graph)
        assert result.status is (
            RetractionSearchStatus.FOUND
            if expected
            else RetractionSearchStatus.EXHAUSTED
        )
        if expected:
            assert result.endomorphism is not None
            assert verify_coaffine_endomorphism(graph, result.endomorphism)
        else:
            assert result.endomorphism is None


def test_coaffine_witness_noninjective_and_labels() -> None:
    graph = nx.empty_graph(["a", "b", "c"])
    assert verify_coaffine_endomorphism(graph, {"a": "b", "b": "a", "c": "a"})
    assert not verify_coaffine_endomorphism(graph, {"a": "a", "b": "a", "c": "a"})
    assert not verify_coaffine_endomorphism(graph, {"a": "b", "b": "a"})
    assert not verify_coaffine_endomorphism(graph, {"a": "b", "b": "a", "c": "d"})
    path = nx.path_graph(4)
    assert not verify_coaffine_endomorphism(path, {0: 2, 1: 3, 2: 0, 3: 1})
    result = find_coaffine_endomorphism(graph)
    assert result.endomorphism is not None
    assert verify_coaffine_endomorphism(graph, result.endomorphism)


def test_coaffine_limits() -> None:
    result = find_coaffine_endomorphism(nx.empty_graph(3), max_states=1)
    assert result.status is RetractionSearchStatus.LIMIT_REACHED
    assert result.endomorphism is None
    assert result.states_explored == 1
    assert find_coaffine_endomorphism(nx.complete_graph(3)).status is (
        RetractionSearchStatus.EXHAUSTED
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_states": 0},
        {"max_states": True},
        {"time_limit": 0},
        {"time_limit": float("nan")},
        {"time_limit": True},
    ],
)
def test_coaffine_invalid_budgets(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        find_coaffine_endomorphism(nx.cycle_graph(4), **kwargs)


def test_proper_stiff_search_matches_brute_force() -> None:
    for graph in nx.graph_atlas_g():
        if len(graph) > 5:
            break
        expected = False
        for order in range(2, len(graph)):
            for target in combinations(graph, order):
                closed = {v: set(graph[v]).intersection(target) | {v} for v in target}
                if any(
                    closed[u] <= closed[v]
                    for u in target
                    for v in target
                    if u != v
                ):
                    continue
                outside = [v for v in graph if v not in target]
                for values in product(target, repeat=len(outside)):
                    mapping = dict(zip(outside, values, strict=True))
                    mapping.update({v: v for v in target})
                    if all(
                        mapping[u] == mapping[v] or graph.has_edge(mapping[u], mapping[v])
                        for u, v in graph.edges()
                    ):
                        expected = True
                        break
                if expected:
                    break
            if expected:
                break
        result = search_proper_stiff_retraction(graph)
        assert result.status is (
            RetractionSearchStatus.FOUND
            if expected
            else RetractionSearchStatus.EXHAUSTED
        )
        if expected:
            assert result.target_vertices is not None
            assert result.retraction is not None
            target_graph = graph.subgraph(result.target_vertices)
            assert verify_retraction_witness(
                graph, target_graph, (result.retraction, {v: v for v in target_graph})
            )
        else:
            assert result.target_vertices is None
            assert result.retraction is None


def test_stiff_search_connected_positive_and_limits() -> None:
    graph = nx.cycle_graph(4)
    graph.add_edge(0, "leaf")
    result = search_proper_stiff_retraction(graph)
    assert result.status is RetractionSearchStatus.FOUND
    assert result.target_vertices is not None
    assert set(result.target_vertices) == {0, 1, 2, 3}
    assert result.retraction is not None
    assert result.retraction["leaf"] == 0 or graph.has_edge(
        0, result.retraction["leaf"]
    )
    for budgets in (
        {"max_subsets": 1},
        {"max_states": 1},
        {"max_assignments": 1, "max_states": 1},
    ):
        limited = search_proper_stiff_retraction(graph, **budgets)
        assert limited.status is RetractionSearchStatus.LIMIT_REACHED
    assert search_proper_stiff_retraction(
        graph, max_target_order=3
    ).status is RetractionSearchStatus.EXHAUSTED
    assert search_proper_stiff_retraction(
        graph, min_target_order=4, max_target_order=4
    ).status is RetractionSearchStatus.FOUND


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_target_order": 1},
        {"max_target_order": 1},
        {"max_subsets": 0},
        {"max_states": 0},
        {"max_assignments": False},
    ],
)
def test_stiff_invalid_budgets(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        search_proper_stiff_retraction(nx.cycle_graph(4), **kwargs)


@pytest.mark.parametrize("graph", [nx.DiGraph(), nx.MultiGraph(), nx.Graph([(0, 0)])])
def test_invalid_graphs(graph: nx.Graph) -> None:
    with pytest.raises(ValueError):
        find_coaffine_endomorphism(graph)
    with pytest.raises(ValueError):
        search_proper_stiff_retraction(graph)
    assert not verify_coaffine_endomorphism(graph, {})


@pytest.mark.parametrize("code", [b"KsX_o_C?_B_M", b"KsX@?oS@_D?J"])
def test_cubic_coaffine_obstruction(code: bytes) -> None:
    graph = completely_pared_graph(nx.complement(nx.from_graph6_bytes(code)))
    for i in range(3):
        result = find_coaffine_endomorphism(graph)
        assert result.status is RetractionSearchStatus.EXHAUSTED
        if i == 0:
            stiff = search_proper_stiff_retraction(graph)
            assert stiff.status is RetractionSearchStatus.EXHAUSTED
        if i < 2:
            nxt = clique_graph(graph)
            assert nxt is not None
            graph = completely_pared_graph(nxt)
