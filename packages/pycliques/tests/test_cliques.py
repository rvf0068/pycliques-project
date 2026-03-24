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


# ---------- More thorough clique_graph tests ----------


def test_clique_graph_of_octahedral():
    """K(octahedron) is the complement of the disjoint union of 4 edges.

    The octahedral graph has 8 triangular cliques.  Its clique graph is the
    complement of 4K2 (four disjoint edges), which is the complete 4-partite
    graph K(2,2,2,2).
    """
    kg = clique_graph(nx.octahedral_graph())
    assert kg is not None
    assert kg.number_of_nodes() == 8
    assert kg.number_of_edges() == 24  # K(2,2,2,2) has C(8,2) - 4 = 24 edges

    complement = nx.complement(kg)
    assert complement.number_of_edges() == 4
    assert all(d == 1 for _, d in complement.degree())


def test_clique_graph_of_complete_graph():
    """K(K_n) is isomorphic to K_1."""
    for n in range(3, 7):
        kg = clique_graph(nx.complete_graph(n))
        assert kg is not None
        assert kg.number_of_nodes() == 1
        assert kg.number_of_edges() == 0


def test_clique_graph_of_cycle():
    """K(C_n) is isomorphic to C_n for n >= 4."""
    for n in range(4, 9):
        kg = clique_graph(nx.cycle_graph(n))
        assert kg is not None
        assert kg.number_of_nodes() == n
        assert kg.number_of_edges() == n
        assert nx.is_isomorphic(kg, nx.cycle_graph(n))


def test_clique_graph_of_path():
    """K(P_n) is isomorphic to P_{n-1} for n >= 2.

    The maximal cliques of a path on n vertices are its n-1 edges.
    """
    for n in range(2, 8):
        kg = clique_graph(nx.path_graph(n))
        assert kg is not None
        assert kg.number_of_nodes() == n - 1
        assert nx.is_isomorphic(kg, nx.path_graph(n - 1))


def test_clique_graph_of_petersen():
    """K(Petersen) has 15 nodes (one per edge) since Petersen is triangle-free."""
    petersen = nx.petersen_graph()
    kg = clique_graph(petersen)
    assert kg is not None
    assert kg.number_of_nodes() == 15  # 15 edges, each is a maximal clique
    assert all(d == 4 for _, d in kg.degree())  # each edge touches 4 others


def test_clique_graph_of_clique_graph_of_petersen():
    """K^2(Petersen) is isomorphic to Petersen since Petersen is triangle-free."""
    petersen = nx.petersen_graph()
    kg = clique_graph(petersen)
    k2g = clique_graph(kg)
    assert k2g is not None
    assert k2g.number_of_nodes() == 10
    assert all(d == 3 for _, d in k2g.degree())


def test_clique_graph_of_empty_graph():
    """K(empty graph on n vertices) has n isolated nodes."""
    g = nx.empty_graph(5)
    kg = clique_graph(g)
    assert kg is not None
    assert kg.number_of_nodes() == 5
    assert kg.number_of_edges() == 0


def test_clique_graph_node_types():
    """All nodes of K(G) are Clique instances."""
    for g in [nx.cycle_graph(5), nx.petersen_graph(), nx.octahedral_graph()]:
        kg = clique_graph(g)
        assert kg is not None
        assert all(isinstance(node, Clique) for node in kg.nodes)


def test_clique_graph_of_complete_bipartite():
    """K(K_{m,n}) is isomorphic to K_m x K_n (tensor product).

    The maximal cliques of K_{m,n} are its m*n edges, and two clique-nodes
    are adjacent iff the edges share an endpoint.  This gives the Kronecker /
    tensor product K_m x K_n (also known as the Rook's graph complement).
    We test via the degree sequence.
    """
    m, n = 3, 4
    g = nx.complete_bipartite_graph(m, n)
    kg = clique_graph(g)
    assert kg is not None
    assert kg.number_of_nodes() == m * n
    # Each edge shares one endpoint with (m-1)+(n-1) others
    expected_degree = (m - 1) + (n - 1)
    assert all(d == expected_degree for _, d in kg.degree())


def test_clique_graph_bound_exact():
    """Bound equal to the exact clique count still succeeds."""
    g = nx.cycle_graph(5)
    assert clique_graph(g, bound=5) is not None
    assert clique_graph(g, bound=4) is None
