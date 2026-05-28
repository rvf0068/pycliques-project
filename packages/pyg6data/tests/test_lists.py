import networkx as nx
import pytest
from pyg6data.lists import (
    cubic_graph_generator,
    digraph_generator,
    graph_generator,
    list_cubic_graphs,
    list_digraphs,
    list_graphs,
    parse_digraph6,
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


# ---------- Tests for digraph6 functions ----------


def test_parse_digraph6_empty_graph():
    """Test parsing the 2-vertex digraph with no edges."""
    g = parse_digraph6("&A?")
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 0


def test_parse_digraph6_single_edge():
    """Test parsing a 2-vertex digraph with one directed edge."""
    g = parse_digraph6("&AO")
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 2
    assert list(g.edges()) == [(0, 1)]


def test_parse_digraph6_bidirectional():
    """Test parsing a 2-vertex digraph with edges in both directions."""
    g = parse_digraph6("&AW")
    assert g.number_of_nodes() == 2
    assert set(g.edges()) == {(0, 1), (1, 0)}


def test_parse_digraph6_single_vertex():
    """Test parsing the trivial 1-vertex digraph."""
    g = parse_digraph6("&@?")
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 1
    assert g.number_of_edges() == 0


def test_parse_digraph6_with_header():
    """Test that the >>digraph6<< header is stripped correctly."""
    g = parse_digraph6(">>digraph6<<&A?")
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 0


def test_parse_digraph6_invalid_prefix():
    """Test that a missing '&' prefix raises ValueError."""
    with pytest.raises(ValueError, match="must start with '&'"):
        parse_digraph6("A?")


def test_parse_digraph6_too_short():
    """Test that a truncated string raises ValueError."""
    with pytest.raises(ValueError, match="too short"):
        # '&B' claims n=3 (9 adjacency bits needed) but provides no data
        parse_digraph6("&B")


def test_digraph_generator_yields_digraphs():
    """Test that digraph_generator yields DiGraph instances of the right size."""
    gen = digraph_generator(2)
    first = next(gen)
    assert isinstance(first, nx.DiGraph)
    assert first.number_of_nodes() == 2


def test_digraph_generator_invalid_input():
    """Test that an unavailable order raises ValueError."""
    with pytest.raises(ValueError, match="is not available"):
        list(digraph_generator(7))


def test_list_digraphs_known_count_n1():
    """Verify exactly 1 non-isomorphic digraph on 1 vertex."""
    digraphs = list_digraphs(1)
    assert len(digraphs) == 1
    assert digraphs[0].number_of_nodes() == 1


def test_list_digraphs_known_count_n2():
    """Verify exactly 3 non-isomorphic digraphs on 2 vertices."""
    digraphs = list_digraphs(2)
    assert len(digraphs) == 3
    assert all(isinstance(g, nx.DiGraph) for g in digraphs)


def test_list_digraphs_known_count_n3():
    """Verify exactly 16 non-isomorphic digraphs on 3 vertices."""
    assert len(list_digraphs(3)) == 16


def test_list_digraphs_known_count_n4():
    """Verify exactly 218 non-isomorphic digraphs on 4 vertices."""
    assert len(list_digraphs(4)) == 218
