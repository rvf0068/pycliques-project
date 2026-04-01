"""Tests for the s_collapses module."""

import networkx as nx
from pycombtop.s_collapses import (
    complete_s_collapse,
    complete_s_collapse_edges,
    has_s_dismantlable_edge,
    has_s_dismantlable_vertex,
    is_s_dismantlable_edge,
    is_s_dismantlable_vertex,
    remove_s_dismantlable_edge,
    remove_s_dismantlable_vertex,
)

# ---------- is_s_dismantlable_vertex ----------


def test_complete_graph_vertex_dismantlable():
    """Every vertex in K_n is s-dismantlable (open neighbourhood is K_{n-1})."""
    g = nx.complete_graph(4)
    assert is_s_dismantlable_vertex(g, 0) is True


def test_cycle_6_vertex_not_dismantlable():
    """No vertex in C_6 is s-dismantlable."""
    g = nx.cycle_graph(6)
    assert is_s_dismantlable_vertex(g, 0) is False


# ---------- has_s_dismantlable_vertex ----------


def test_has_vertex_complete():
    """K_n has an s-dismantlable vertex."""
    g = nx.complete_graph(4)
    assert has_s_dismantlable_vertex(g) is not None


def test_has_vertex_cycle_6():
    """C_6 has no s-dismantlable vertices."""
    g = nx.cycle_graph(6)
    assert has_s_dismantlable_vertex(g) is None


# ---------- remove_s_dismantlable_vertex ----------


def test_remove_vertex_reduces_order():
    """Removing an s-dismantlable vertex decreases order by 1."""
    g = nx.complete_graph(5)
    g2 = remove_s_dismantlable_vertex(g)
    assert g2.order() == 4


def test_remove_vertex_no_change():
    """When no vertex is dismantlable, graph is unchanged."""
    g = nx.cycle_graph(6)
    g2 = remove_s_dismantlable_vertex(g)
    assert g2.order() == 6


# ---------- complete_s_collapse ----------


def test_complete_collapse_complete_graph():
    """K_n collapses to a single vertex."""
    g = complete_s_collapse(nx.complete_graph(5))
    assert g.order() == 1


def test_complete_collapse_cycle_6():
    """C_6 cannot be reduced (no s-dismantlable vertices)."""
    g = complete_s_collapse(nx.cycle_graph(6))
    assert g.order() == 6


def test_complete_collapse_preserves_cycle_5():
    """C_5 has no s-dismantlable vertices."""
    g = complete_s_collapse(nx.cycle_graph(5))
    assert g.order() == 5


# ---------- is_s_dismantlable_edge ----------


def test_triangle_edge_dismantlable():
    """In K_3, every edge is s-dismantlable (common neighbour is a single vertex)."""
    g = nx.complete_graph(3)
    assert is_s_dismantlable_edge(g, (0, 1)) is True


def test_cycle_4_edge_not_dismantlable():
    """In C_4, no edge is s-dismantlable (common neighbours are empty)."""
    g = nx.cycle_graph(4)
    assert is_s_dismantlable_edge(g, (0, 1)) is False


# ---------- has_s_dismantlable_edge ----------


def test_has_edge_triangle():
    """K_3 has an s-dismantlable edge."""
    assert has_s_dismantlable_edge(nx.complete_graph(3)) is not None


def test_has_edge_cycle_4():
    """C_4 has no s-dismantlable edges."""
    assert has_s_dismantlable_edge(nx.cycle_graph(4)) is None


# ---------- remove_s_dismantlable_edge ----------


def test_remove_edge_reduces_size():
    """Removing an s-dismantlable edge decreases size by 1."""
    g = nx.complete_graph(3)
    g2 = remove_s_dismantlable_edge(g)
    assert g2.size() == 2


def test_remove_edge_no_change():
    """When no edge is dismantlable, graph is unchanged."""
    g = nx.cycle_graph(4)
    g2 = remove_s_dismantlable_edge(g)
    assert g2.size() == 4


# ---------- complete_s_collapse_edges ----------


def test_complete_collapse_edges_k4():
    """K_4 edge-collapses from 6 edges to 3 (a cycle)."""
    g = complete_s_collapse_edges(nx.complete_graph(4))
    assert g.size() == 3


def test_complete_collapse_edges_cycle_4():
    """C_4 has no s-dismantlable edges; unchanged."""
    g = complete_s_collapse_edges(nx.cycle_graph(4))
    assert g.size() == 4
