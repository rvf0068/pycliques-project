"""Check the face-free bound against exact binary homology."""

import networkx as nx
import pytest

from pycombtop import recursive_betti_upper_bounds
from pycombtop.bounded_betti import bounded_reduced_betti_mod2


def test_all_small_graphs_against_exact_homology():
    for graph in nx.graph_atlas_g():
        if len(graph) > 5:
            continue
        result = recursive_betti_upper_bounds(graph, max_degree=4)
        exact = bounded_reduced_betti_mod2(graph, max_degree=4)
        assert not result.is_limited
        assert exact is not None
        assert all(
            result.upper_bounds[d] >= exact.get(d, 0) for d in range(5)
        )
        assert all(exact.get(d, 0) == 0 for d in result.known_zero_betti)


def test_octahedron_and_cone():
    graph = nx.complement(nx.Graph([(0, 1), (2, 3), (4, 5), (6, 7)]))
    result = recursive_betti_upper_bounds(graph, max_degree=4)
    assert result.upper_bounds == {0: 0, 1: 0, 2: 0, 3: 1, 4: 0}
    graph.add_edges_from((8, v) for v in range(8))
    assert recursive_betti_upper_bounds(graph, max_degree=4).known_zero_betti == dict.fromkeys(range(5), 0)


def test_limit_keeps_only_completed_degrees():
    graph = nx.empty_graph(4)
    result = recursive_betti_upper_bounds(graph, max_degree=3, max_states=1)
    assert result.is_limited
    assert result.states_explored == 1
    assert result.upper_bounds == {0: 3}
    assert not result.known_zero_betti


def test_arbitrary_labels_and_cache():
    graph = nx.cycle_graph(["a", "b", "c", "d", "e", "f"])
    cached = recursive_betti_upper_bounds(graph, max_degree=3)
    uncached = recursive_betti_upper_bounds(graph, max_degree=3, max_cache=0)
    assert cached.upper_bounds == uncached.upper_bounds


@pytest.mark.parametrize(
    "kwargs",
    [{"max_degree": -1}, {"max_states": 0}, {"max_cache": -1}, {"max_degree": True}],
)
def test_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        recursive_betti_upper_bounds(nx.path_graph(2), **({"max_degree": 2} | kwargs))


def test_invalid_graph():
    with pytest.raises(ValueError):
        recursive_betti_upper_bounds(nx.DiGraph(), max_degree=1)
