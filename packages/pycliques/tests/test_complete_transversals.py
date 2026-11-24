"""Complete-transversal search against explicit maximal-clique enumeration."""

from itertools import combinations

import networkx as nx
import pytest
from pycliques import (
    RetractionSearchStatus,
    find_complete_transversal,
    find_split_cover_counterexample,
    verify_complete_transversal,
)


def test_complete_transversal_matches_small_oracle() -> None:
    for graph in nx.graph_atlas_g():
        if len(graph) > 4:
            break
        subsets = [
            frozenset(s) for k in range(len(graph) + 1) for s in combinations(graph, k)
        ]
        cliques = [set(q) for q in nx.find_cliques(graph)]
        for a in subsets:
            for b in subsets:
                family = [a, b]
                expected = any(all(q & s for s in family) for q in cliques)
                result = find_complete_transversal(graph, family)
                assert result.status is (
                    RetractionSearchStatus.FOUND
                    if expected
                    else RetractionSearchStatus.EXHAUSTED
                )
                if expected:
                    assert result.transversal is not None
                    assert verify_complete_transversal(
                        graph, family, result.transversal
                    )
                else:
                    assert result.transversal is None


def test_empty_family_and_labels() -> None:
    graph = nx.path_graph(["a", "b", "c"])
    assert find_complete_transversal(graph, []).transversal == {"a"}
    assert find_complete_transversal(nx.Graph(), []).status is (
        RetractionSearchStatus.EXHAUSTED
    )
    assert find_complete_transversal(graph, [[]]).status is (
        RetractionSearchStatus.EXHAUSTED
    )
    assert not verify_complete_transversal(graph, [], [])
    assert not verify_complete_transversal(graph, [["c"]], ["a"])
    assert not verify_complete_transversal(graph, [], ["a", "c"])
    assert not verify_complete_transversal(graph, [], ["absent"])
    assert not verify_complete_transversal(graph, [["absent"]], ["a"])


def test_limits_and_validation() -> None:
    graph = nx.path_graph(3)
    result = find_complete_transversal(graph, [{0}, {1}], max_states=1)
    assert result.status is RetractionSearchStatus.LIMIT_REACHED
    assert result.transversal is None
    assert result.states_explored == 1
    for bound in (0, True):
        with pytest.raises(ValueError):
            find_complete_transversal(graph, [], max_states=bound)
    with pytest.raises(ValueError):
        find_complete_transversal(graph, [{5}])
    with pytest.raises(ValueError):
        find_complete_transversal(nx.DiGraph(), [])


def test_split_cover_checks() -> None:
    graph = nx.path_graph(3)
    splits = [({0, 1}, {2}), ({0}, {1, 2})]
    assert find_split_cover_counterexample(graph, splits).status is (
        RetractionSearchStatus.EXHAUSTED
    )
    result = find_split_cover_counterexample(graph, splits[:1])
    assert result.status is RetractionSearchStatus.FOUND
    assert result.transversal == {1, 2}
    for bad in [({0}, {2}), ({0, 1}, {1, 2}), ({0, 1}, {2, 3})]:
        with pytest.raises(ValueError):
            find_split_cover_counterexample(graph, [bad])
    assert find_split_cover_counterexample(graph, [(set(graph), set())]).status is (
        RetractionSearchStatus.EXHAUSTED
    )
