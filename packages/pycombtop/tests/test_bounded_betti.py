"""Independent bounded GF(2) Betti consistency and resource tests."""

import networkx as nx
import pytest
from pycliques.named import complement_of_cycle, octahedron, snub_disphenoid
from pycombtop.bounded_betti import bounded_reduced_betti_mod2
from pycombtop.homotopy_type import betti_numbers_graph


@pytest.mark.parametrize("graph",[
    nx.path_graph(5),
    nx.cycle_graph(5),
    octahedron(3),
    octahedron(4),
    complement_of_cycle(8),
    complement_of_cycle(9),
    snub_disphenoid(),
])
def test_full_vector_agrees_with_existing_homology(graph):
    ranks = bounded_reduced_betti_mod2(graph)
    assert ranks is not None
    expected = betti_numbers_graph(graph)
    for degree, value in ranks.items():
        assert value == (expected[degree] if degree < len(expected) else 0)


def test_partial_homology_does_not_infer_uncomputed_degrees():
    graph = octahedron(4)
    full = bounded_reduced_betti_mod2(graph)
    partial = bounded_reduced_betti_mod2(graph,max_degree=2)
    assert full is not None and partial is not None
    assert full[3]==1 and 3 not in partial
    assert partial == {0:0,1:0,2:0}


def test_budget_exceeding_returns_unknown_not_zero():
    assert bounded_reduced_betti_mod2(
        octahedron(3),max_faces=3,
    ) is None
    assert bounded_reduced_betti_mod2(
        octahedron(3),max_faces=3,max_degree=2,
    ) is None


def test_disconnected_complex_reduced_beta_zero():
    ranks=bounded_reduced_betti_mod2(nx.empty_graph(4))
    assert ranks == {0:3}


def test_invalid_limits():
    with pytest.raises(ValueError):
        bounded_reduced_betti_mod2(nx.cycle_graph(5),max_faces=0)
    with pytest.raises(ValueError):
        bounded_reduced_betti_mod2(nx.cycle_graph(5),max_degree=-1)
