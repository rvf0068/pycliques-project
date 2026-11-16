"""Independent bounded GF(2) Betti consistency and resource tests."""

import networkx as nx
import pytest
from pycliques.cliques import clique_graph
from pycliques.named import complement_of_cycle, octahedron, snub_disphenoid
from pycombtop.bounded_betti import bounded_reduced_betti_mod2


@pytest.mark.parametrize(
    ("graph", "nonzero"),
    [
        (nx.path_graph(5), {}),
        (nx.cycle_graph(5), {1: 1}),
        (octahedron(3), {2: 1}),
        (octahedron(4), {3: 1}),
        (complement_of_cycle(8), {2: 1}),
        (complement_of_cycle(9), {2: 2}),
        (snub_disphenoid(), {2: 1}),
    ],
)
def test_full_vector_matches_independent_known_homology(graph, nonzero):
    ranks = bounded_reduced_betti_mod2(graph)
    assert ranks is not None
    assert {d: rank for d, rank in ranks.items() if rank} == nonzero


def test_snub_k1_k2_regression_against_exact_boundary_calculation():
    graph = snub_disphenoid()
    for expected_order, expected_edges, max_dimension in (
        (12, 46, 4),
        (20, 136, 6),
    ):
        graph = clique_graph(graph, bound=1200)
        assert graph is not None
        assert (len(graph), graph.number_of_edges()) == (
            expected_order, expected_edges,
        )
        ranks = bounded_reduced_betti_mod2(graph)
        assert ranks is not None
        assert ranks == {
            dimension: (1 if dimension == 2 else 0)
            for dimension in range(max_dimension + 1)
        }


def test_partial_homology_does_not_infer_uncomputed_degrees():
    graph = octahedron(4)
    full = bounded_reduced_betti_mod2(graph)
    partial = bounded_reduced_betti_mod2(graph, max_degree=2)
    assert full is not None and partial is not None
    assert full[3] == 1 and 3 not in partial
    assert partial == {0: 0, 1: 0, 2: 0}


def test_budget_exceeding_returns_unknown_not_zero():
    assert (
        bounded_reduced_betti_mod2(
            octahedron(3),
            max_faces=3,
        )
        is None
    )
    assert (
        bounded_reduced_betti_mod2(
            octahedron(3),
            max_faces=3,
            max_degree=2,
        )
        is None
    )


def test_disconnected_complex_reduced_beta_zero():
    ranks = bounded_reduced_betti_mod2(nx.empty_graph(4))
    assert ranks == {0: 3}


def test_empty_graph_homology():
    assert bounded_reduced_betti_mod2(nx.Graph()) == {}


def test_invalid_limits():
    with pytest.raises(ValueError):
        bounded_reduced_betti_mod2(nx.cycle_graph(5), max_faces=0)
    with pytest.raises(ValueError):
        bounded_reduced_betti_mod2(nx.cycle_graph(5), max_degree=-1)
