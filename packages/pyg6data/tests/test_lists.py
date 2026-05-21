import networkx as nx
import pytest
from pyg6data.lists import (
    cubic_graph_generator,
    graph_generator,
    list_cubic_graphs,
    list_graphs,
    small_torsion_graphs,
)


def test_graph_generator_valid_input():
    """Test that the generator successfully yields NetworkX graphs."""
    # Grab just the first graph from n=6 connected
    gen = graph_generator(n=6, connected=True)
    first_graph = next(gen)

    assert isinstance(first_graph, nx.Graph)
    assert first_graph.number_of_nodes() == 6
    assert nx.is_connected(first_graph)


def test_graph_generator_invalid_input():
    """Test that asking for unavailable data raises a ValueError."""
    with pytest.raises(ValueError, match="is not available"):
        # n=5 is not in our dictionaries
        list(graph_generator(n=5, connected=True))


def test_list_graphs_known_count():
    """Verify that we load the exact number of expected graphs for n=6."""
    # Brendan McKay's data confirms there are exactly 112 connected graphs on
    # 6 vertices
    graphs_n6_connected = list_graphs(n=6, connected=True)

    assert isinstance(graphs_n6_connected, list)
    assert len(graphs_n6_connected) == 112

    # Just to be thorough, verify the first one is actually a graph
    assert isinstance(graphs_n6_connected[0], nx.Graph)


def test_small_torsion_graphs():
    """Test that the small torsion uncompressed .g6 file loads correctly."""
    torsion_graphs = small_torsion_graphs()

    assert isinstance(torsion_graphs, list)
    assert len(torsion_graphs) > 0
    assert isinstance(torsion_graphs[0], nx.Graph)


# ---------- Coverage for non-gzip file reading (lines 75-79) ----------


def test_graph_generator_non_gzip():
    """Test the non-gzip file path (e.g., graph5.g6)."""
    graphs = list(graph_generator(n=5, connected=False))
    assert len(graphs) > 0
    assert all(isinstance(g, nx.Graph) for g in graphs)
    assert all(g.number_of_nodes() == 5 for g in graphs)


# ---------- Tests for cubic graph functions ----------


def test_cubic_graph_generator_valid_input():
    """Test that the cubic generator yields valid cubic connected graphs."""
    gen = cubic_graph_generator(n=8)
    first_graph = next(gen)

    assert isinstance(first_graph, nx.Graph)
    assert first_graph.number_of_nodes() == 8
    assert nx.is_connected(first_graph)
    assert all(d == 3 for _, d in first_graph.degree())


def test_cubic_graph_generator_invalid_input():
    """Test that asking for an unavailable order raises a ValueError."""
    with pytest.raises(ValueError, match="is not available"):
        list(cubic_graph_generator(n=7))


def test_list_cubic_graphs_known_count_n8():
    """Verify the exact count of cubic connected graphs on 8 vertices."""
    # There are exactly 5 cubic connected graphs on 8 vertices (OEIS A002851)
    graphs = list_cubic_graphs(n=8)

    assert isinstance(graphs, list)
    assert len(graphs) == 5
    assert all(isinstance(g, nx.Graph) for g in graphs)


def test_list_cubic_graphs_known_count_n10():
    """Verify the exact count of cubic connected graphs on 10 vertices."""
    # There are exactly 19 cubic connected graphs on 10 vertices (OEIS A002851)
    graphs = list_cubic_graphs(n=10)

    assert len(graphs) == 19


def test_cubic_graph_generator_gzip():
    """Test the gzip file path for cubic graphs (e.g., cub18.g6.gz)."""
    gen = cubic_graph_generator(n=18)
    first_graph = next(gen)

    assert isinstance(first_graph, nx.Graph)
    assert first_graph.number_of_nodes() == 18
    assert all(d == 3 for _, d in first_graph.degree())
