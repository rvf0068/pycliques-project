"""Regression tests for face-free homotopy classification."""

import networkx as nx
import pytest
from pycombtop import (
    Theorem,
    connected_link_order,
    homotopy_type_with_verdict,
    recursive_betti_upper_bounds,
    verify_connected_link_order,
)
from pycombtop.bounded_betti import bounded_reduced_betti_mod2
from pycombtop.homotopy_type import _try_recursive_betti_wedge


def test_all_degree_mode_matches_scalar_and_euler():
    for graph in nx.graph_atlas_g():
        if len(graph) > 5:
            break
        simultaneous = recursive_betti_upper_bounds(graph)
        scalar = recursive_betti_upper_bounds(graph, max_degree=4)
        assert not simultaneous.is_limited
        assert all(
            simultaneous.upper_bounds.get(d, 0) == b
            for d, b in scalar.upper_bounds.items()
        )
        chi = -1 + sum(
            (-1) ** (len(face) - 1) for face in nx.enumerate_all_cliques(graph)
        )
        assert simultaneous.reduced_euler_characteristic == chi
        exact = bounded_reduced_betti_mod2(graph, max_degree=4)
        assert exact is not None
        assert all(
            exact.get(d, 0) <= b <= simultaneous.upper_bounds[d]
            for d, b in simultaneous.euler_refined_upper_bounds.items()
        )
        for policy in ("ascending", "reverse_ties"):
            alternate = recursive_betti_upper_bounds(graph, neighbor_order=policy)
            assert alternate.reduced_euler_characteristic == chi
            assert all(
                b >= exact.get(d, 0)
                for d, b in alternate.euler_refined_upper_bounds.items()
            )


def test_all_degree_limit_is_not_vanishing():
    result = recursive_betti_upper_bounds(nx.path_graph(5), max_states=1)
    assert result.is_limited
    assert result.upper_bounds == {}
    assert result.reduced_euler_characteristic is None


def test_connected_link_order_and_replay():
    graph = nx.complete_multipartite_graph(2, 2, 2)
    graph = nx.relabel_nodes(graph, lambda v: ("vertex", v))
    order = connected_link_order(graph)
    assert order is not None
    assert verify_connected_link_order(graph, order)
    assert not verify_connected_link_order(graph, order[:-1])
    assert not verify_connected_link_order(graph, order[:-1] + (order[0],))
    cycle = nx.cycle_graph(5)
    assert connected_link_order(cycle) is None
    assert not verify_connected_link_order(cycle, tuple(cycle))
    assert connected_link_order(nx.empty_graph(2)) is None
    assert connected_link_order(nx.Graph()) is None


@pytest.mark.parametrize("size", [3, 5])
def test_hurewicz_wedge_and_cone(size):
    sphere = nx.complete_multipartite_graph(*([2] * size))
    verdict = _try_recursive_betti_wedge(sphere, 1000)
    assert verdict is not None and verdict.is_exact
    assert verdict.wedge.spheres == {size - 1: 1}
    sphere.add_edges_from((2 * size, v) for v in range(2 * size))
    verdict = _try_recursive_betti_wedge(sphere, 1000)
    assert verdict is not None and verdict.wedge.spheres == {}


def test_hurewicz_does_not_claim_circle_or_limit():
    assert _try_recursive_betti_wedge(nx.cycle_graph(5), 1000) is None
    assert _try_recursive_betti_wedge(nx.cycle_graph(5), 1) is None
    assert _try_recursive_betti_wedge(nx.empty_graph(4), 1000) is None
    with pytest.raises(ValueError):
        recursive_betti_upper_bounds(nx.path_graph(3), neighbor_order="invalid")
    with pytest.raises(ValueError):
        connected_link_order(nx.path_graph(3), max_roots=0)
    with pytest.raises(ValueError):
        homotopy_type_with_verdict(nx.path_graph(3), recursive_betti_states=-1)


def test_classifier_face_free_cubic_regression():
    from pycliques import CliqueSequence

    cubic = nx.from_graph6_bytes(b"M{S__SE@?A?C?F?F?")
    sequence = CliqueSequence(
        nx.complement(cubic), bound=1000, relabel=True, pare_initial=True
    )
    first = sequence[1]
    assert first is not None
    verdict = homotopy_type_with_verdict(first, max_simplices=0)
    assert verdict.is_exact and verdict.wedge.spheres == {}
    assert verdict.reason == Theorem.RECURSIVE_BETTI_HUREWICZ
