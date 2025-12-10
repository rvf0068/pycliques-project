import networkx as nx

from pycliques import Clique, clique_graph, homotopy_clique_graph


def test_clique_creation():
    """Test that we can create a clique from a list or set."""
    c = Clique([1, 2, 3])
    assert 1 in c
    assert len(c) == 3
    assert isinstance(c, frozenset)


def test_clique_repr():
    """Test the custom string representation."""
    c = Clique([1, 2])
    # The set order isn't guaranteed, so we check both possibilities
    assert repr(c) == "{1, 2}" or repr(c) == "{2, 1}"


def test_empty_clique_repr():
    """Test the representation of an empty clique."""
    c = Clique([])
    assert repr(c) == "{}"


def test_clique_equality():
    """Test that Cliques compare equal to standard sets/frozensets."""
    c = Clique([1, 2])
    assert c == {1, 2}
    assert c == frozenset([1, 2])


def test_clique_graph_returns_clique_nodes():
    """Clique graph should output Clique nodes for a simple square graph."""
    graph = nx.cycle_graph(4)
    result = clique_graph(graph)
    assert result is not None
    assert len(result.nodes) == 4
    assert all(isinstance(node, Clique) for node in result.nodes)


def test_clique_graph_bound_returns_none_when_exceeded():
    """If the bound is too small, the computation aborts with None."""
    graph = nx.cycle_graph(4)
    assert clique_graph(graph, bound=2) is None


def test_homotopy_clique_graph_vertex_clique_pairs():
    """Nodes in the homotopy clique graph are (vertex, clique) pairs."""
    graph = nx.path_graph(3)
    h_graph = homotopy_clique_graph(graph)
    assert len(h_graph) == 4
    for vertex, clique in h_graph.nodes:
        assert isinstance(vertex, int)
        assert isinstance(clique, Clique)
        assert vertex in clique


def test_homotopy_clique_graph_preserves_connectivity_on_path():
    """H(P3) should be connected because all nodes share adjacency."""
    graph = nx.path_graph(3)
    h_graph = homotopy_clique_graph(graph)
    assert nx.is_connected(h_graph)
