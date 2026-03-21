import networkx as nx
import pytest
from pyg6data.lists import graph_generator, list_graphs, small_torsion_graphs


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
