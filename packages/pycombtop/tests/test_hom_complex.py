"""Tests for pycombtop.hom_complex."""

import networkx as nx
from pycombtop.hom_complex import graph_homomorphisms, hom_graph

# ---------------------------------------------------------------------------
# graph_homomorphisms
# ---------------------------------------------------------------------------


def test_homomorphisms_k2_to_k2():
    """K2 → K2: only the two edge-preserving dicts are valid."""
    g = nx.complete_graph(2)
    h = nx.complete_graph(2)
    homs = graph_homomorphisms(g, h)
    assert len(homs) == 2
    assert all(isinstance(f, dict) for f in homs)
    assert {0: 0, 1: 1} in homs
    assert {0: 1, 1: 0} in homs


def test_homomorphisms_k1_to_kn():
    """K1 → Kn: every vertex of H is a valid image."""
    n = 4
    g = nx.complete_graph(1)
    h = nx.complete_graph(n)
    homs = graph_homomorphisms(g, h)
    assert len(homs) == n
    assert all(len(f) == 1 for f in homs)


def test_homomorphisms_k3_to_k2_empty():
    """K3 is 3-chromatic, so no homomorphism to K2 exists."""
    g = nx.complete_graph(3)
    h = nx.complete_graph(2)
    assert graph_homomorphisms(g, h) == []


def test_homomorphisms_path3_to_k2():
    """Path P3 is bipartite and 2-colorable, so homomorphisms to K2 exist."""
    g = nx.path_graph(3)  # 0-1-2
    h = nx.complete_graph(2)
    homs = graph_homomorphisms(g, h)
    # Valid colorings of P3 with 2 colors: 0→0,1→1,2→0 and 0→1,1→0,2→1
    assert len(homs) == 2
    assert all(len(f) == 3 for f in homs)


def test_homomorphisms_preserve_edges():
    """Each returned mapping must send every edge of G to an edge of H."""
    g = nx.cycle_graph(4)
    h = nx.complete_graph(3)
    for f in graph_homomorphisms(g, h):
        for u, v in g.edges():
            assert h.has_edge(f[u], f[v]), f"Edge ({u},{v}) not preserved by {f}"


def test_homomorphisms_string_vertices():
    """Homomorphisms work correctly with non-integer vertex labels."""
    g = nx.Graph()
    g.add_edge("a", "b")
    h = nx.Graph()
    h.add_edge("x", "y")
    homs = graph_homomorphisms(g, h)
    assert len(homs) == 2
    assert all(isinstance(f, dict) for f in homs)
    assert {"a": "x", "b": "y"} in homs
    assert {"a": "y", "b": "x"} in homs


# ---------------------------------------------------------------------------
# hom_graph – node counts
# ---------------------------------------------------------------------------


def test_hom_graph_single_vertex_domain_is_complete():
    """Hom(K1, Kn) should be the complete graph Kn."""
    n = 5
    g = nx.complete_graph(1)
    h = nx.complete_graph(n)
    result = hom_graph(g, h)
    assert result.number_of_nodes() == n
    assert result.number_of_edges() == n * (n - 1) // 2


def test_hom_graph_k2_to_k2_no_edges():
    """Hom(K2, K2) has 2 nodes and 0 edges."""
    result = hom_graph(nx.complete_graph(2), nx.complete_graph(2))
    assert result.number_of_nodes() == 2
    assert result.number_of_edges() == 0


def test_hom_graph_no_homomorphisms():
    """When no homomorphism exists the result graph is empty."""
    result = hom_graph(nx.complete_graph(3), nx.complete_graph(2))
    assert result.number_of_nodes() == 0
    assert result.number_of_edges() == 0


def test_hom_graph_k2_to_k3_is_6_cycle():
    """Hom(K2, K3) is the 6-cycle C6 (a classical result)."""
    result = hom_graph(nx.complete_graph(2), nx.complete_graph(3))
    assert result.number_of_nodes() == 6
    assert result.number_of_edges() == 6
    assert all(deg == 2 for _, deg in result.degree())


def test_hom_graph_returns_graph_instance():
    """The return type must be a networkx Graph."""
    result = hom_graph(nx.path_graph(2), nx.complete_graph(3))
    assert isinstance(result, nx.Graph)


def test_hom_graph_nodes_are_frozensets():
    """Nodes of hom_graph must be frozensets convertible to dicts."""
    result = hom_graph(nx.path_graph(2), nx.complete_graph(3))
    for node in result.nodes():
        assert isinstance(node, frozenset)
        f = dict(node)
        assert isinstance(f, dict)


# ---------------------------------------------------------------------------
# hom_graph – adjacency condition
# ---------------------------------------------------------------------------


def test_hom_graph_adjacency_implies_compatibility():
    """For every edge in result, the compatibility condition must hold."""
    g = nx.path_graph(3)
    h = nx.complete_graph(3)
    result = hom_graph(g, h)

    for f_node, phi_node in result.edges():
        f, phi = dict(f_node), dict(phi_node)
        for u, v in g.edges():
            assert h.has_edge(f[u], phi[v])
            assert h.has_edge(f[v], phi[u])


def test_hom_graph_non_adjacent_fails_condition():
    """For non-adjacent pairs, at least one compatibility check must fail."""
    g = nx.complete_graph(2)
    h = nx.complete_graph(3)
    result = hom_graph(g, h)

    all_homs = list(result.nodes())
    for i, f_node in enumerate(all_homs):
        for phi_node in all_homs[i + 1 :]:
            if not result.has_edge(f_node, phi_node):
                f, phi = dict(f_node), dict(phi_node)
                failed = any(
                    not h.has_edge(f[u], phi[v]) or not h.has_edge(f[v], phi[u])
                    for u, v in g.edges()
                )
                assert failed, f"Non-adjacent pair {f}, {phi} passed all checks"


# ---------------------------------------------------------------------------
# hom_graph – nodes are homomorphisms
# ---------------------------------------------------------------------------


def test_hom_graph_nodes_are_valid_homomorphisms():
    """Every node in Hom(G, H) must be a valid homomorphism from G to H."""
    g = nx.cycle_graph(4)
    h = nx.complete_graph(3)
    result = hom_graph(g, h)

    for node in result.nodes():
        f = dict(node)
        for u, v in g.edges():
            assert h.has_edge(f[u], f[v]), f"Node {f} is not a valid homomorphism"


def test_hom_graph_node_count_matches_homomorphism_count():
    """The number of nodes equals the number of graph homomorphisms."""
    g = nx.path_graph(3)
    h = nx.complete_graph(4)
    homs = graph_homomorphisms(g, h)
    result = hom_graph(g, h)
    assert result.number_of_nodes() == len(homs)
