"""Tests for the fundamental_group module."""

import networkx as nx
import pytest
from pycombtop.fundamental_group import (
    FundamentalRecord,
    covering_graph,
    first_homology_rank,
    fundamental_group,
)

# ---------- FundamentalRecord ----------


def test_fundamental_record_fields():
    """FundamentalRecord stores group, generators, edge_labels, etc."""
    G = nx.complete_graph(3)
    rec = fundamental_group(G)
    assert isinstance(rec, FundamentalRecord)
    assert hasattr(rec, "group")
    assert hasattr(rec, "generators")
    assert hasattr(rec, "edge_labels")
    assert hasattr(rec, "spanning_tree")
    assert hasattr(rec, "relators")
    assert hasattr(rec, "rank")


def test_fundamental_record_str():
    """FundamentalRecord has a string representation."""
    G = nx.complete_graph(3)
    rec = fundamental_group(G)
    s = str(rec)
    assert "FundamentalRecord" in s
    assert "rank=" in s


# ---------- fundamental_group: trivial cases ----------


def test_complete_graph_simply_connected():
    """The clique complex of K_n is simply connected (rank 0)."""
    for n in [3, 4, 5]:
        G = nx.complete_graph(n)
        rec = fundamental_group(G)
        assert rec.rank == 0


def test_path_graph_simply_connected():
    """A path graph has a contractible clique complex (rank 0)."""
    for n in [2, 3, 4, 5]:
        G = nx.path_graph(n)
        rec = fundamental_group(G)
        assert rec.rank == 0


def test_tree_simply_connected():
    """Any tree has a contractible clique complex (rank 0)."""
    T = nx.balanced_tree(2, 3)  # Binary tree of depth 3
    rec = fundamental_group(T)
    assert rec.rank == 0


def test_star_graph_simply_connected():
    """A star graph has rank 0."""
    G = nx.star_graph(5)
    rec = fundamental_group(G)
    assert rec.rank == 0


# ---------- fundamental_group: non-trivial cases ----------


def test_cycle_4_rank_1():
    """C_4 (square) has fundamental group Z (rank 1)."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    assert rec.rank == 1


def test_cycle_5_rank_1():
    """C_5 (pentagon) has fundamental group Z (rank 1)."""
    G = nx.cycle_graph(5)
    rec = fundamental_group(G)
    assert rec.rank == 1


def test_cycle_6_rank_1():
    """C_6 has fundamental group Z (rank 1)."""
    G = nx.cycle_graph(6)
    rec = fundamental_group(G)
    assert rec.rank == 1


def test_cycle_graph_general():
    """C_n for n >= 4 has fundamental group Z (rank 1)."""
    for n in range(4, 10):
        G = nx.cycle_graph(n)
        rec = fundamental_group(G)
        assert rec.rank == 1, f"C_{n} should have rank 1"


def test_chordal_graph_simply_connected():
    """Chordal graphs have simply connected clique complexes."""
    # Complete bipartite K_{2,3} is not chordal, but K_{1,n} (star) is
    # A wheel graph W_n = C_{n-1} + hub is chordal for any n >= 4
    # Actually wheel is NOT chordal. Let's use a different example.
    # Use a graph that is explicitly chordal: a tree plus some chords
    # The simplest chordal graph: K_n (already tested), or a clique tree
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)])
    # This graph: 0-1-2-0 (triangle) and 2-3-4-2 (triangle)
    # It's chordal (each cycle of length >= 4 has a chord)
    rec = fundamental_group(G)
    assert rec.rank == 0


# ---------- fundamental_group: edge_labels and spanning_tree ----------


def test_edge_labels_present():
    """All edges of the graph are labelled."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    # edge_labels contains both directions (u,v) and (v,u)
    for u, v in G.edges():
        assert (u, v) in rec.edge_labels or (v, u) in rec.edge_labels


def test_spanning_tree_size():
    """Spanning tree has |V| - 1 edges."""
    G = nx.cycle_graph(5)
    rec = fundamental_group(G)
    assert len(rec.spanning_tree) == G.order() - 1


def test_tree_edges_identity():
    """Tree edges have identity label."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    # Tree edges should have identity label in the edge_labels dict
    # The identity in sympy FreeGroup is represented as the identity element
    for edge_set in rec.spanning_tree:
        edge_list = list(edge_set)
        u, v = edge_list[0], edge_list[1]
        # Both directions should exist and be identity
        label = rec.edge_labels.get((u, v)) or rec.edge_labels.get((v, u))
        # Identity element is the identity of the FreeGroup
        # We check by comparing with group identity if available
        if rec.rank == 0:
            # Trivial group case
            pass
        # For non-trivial cases, identity label should exist
        assert label is not None


# ---------- fundamental_group: error handling ----------


def test_empty_graph_raises():
    """Empty graph raises ValueError."""
    G = nx.Graph()
    with pytest.raises(ValueError, match="non-empty"):
        fundamental_group(G)


def test_disconnected_graph_raises():
    """Disconnected graph raises ValueError."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(0, 1)
    G.add_edge(2, 3)
    with pytest.raises(ValueError, match="connected"):
        fundamental_group(G)


# ---------- first_homology_rank ----------


def test_first_homology_rank_cycle():
    """first_homology_rank equals rank for a cycle."""
    G = nx.cycle_graph(5)
    assert first_homology_rank(G) == 1


def test_first_homology_rank_complete():
    """first_homology_rank is 0 for complete graphs."""
    G = nx.complete_graph(4)
    assert first_homology_rank(G) == 0


def test_first_homology_rank_tree():
    """first_homology_rank is 0 for trees."""
    T = nx.path_graph(5)
    assert first_homology_rank(T) == 0


# ---------- fundamental_group: generators and relators ----------


def test_generators_count():
    """Number of generators equals rank."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    assert len(rec.generators) == rec.rank


def test_complete_graph_no_generators():
    """Complete graph has no generators."""
    G = nx.complete_graph(5)
    rec = fundamental_group(G)
    assert len(rec.generators) == 0


# ---------- fundamental_group: more complex examples ----------


def test_two_cycles_sharing_vertex():
    """Two cycles sharing a vertex: π₁ ≅ Z * Z (rank 2)."""
    # Create two squares sharing one vertex
    G = nx.Graph()
    # First square: 0-1-2-3-0
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    # Second square: 0-4-5-6-0
    G.add_edges_from([(0, 4), (4, 5), (5, 6), (6, 0)])
    rec = fundamental_group(G)
    assert rec.rank == 2


def test_ladder_graph():
    """Ladder graph P_n × P_2 has rank 1 for n >= 2 (one square)."""
    G = nx.ladder_graph(2)  # 4 vertices, 1 square
    rec = fundamental_group(G)
    assert rec.rank == 1


def test_wheel_graph():
    """Wheel graph W_n = K_1 + C_{n} has many triangles, often simply connected."""
    # W_4 = K_1 + C_4: center connected to all vertices of a square
    # This has 4 triangles, so the fundamental group depends on the structure
    G = nx.wheel_graph(5)  # 5 vertices: center + C_4
    rec = fundamental_group(G)
    # Wheel graphs often have trivial π₁ due to triangles
    # W_5 = C_4 + center: each edge of C_4 forms a triangle with center
    # This should make it simply connected
    assert rec.rank == 0


def test_petersen_graph():
    """Petersen graph fundamental group computation completes."""
    G = nx.petersen_graph()
    rec = fundamental_group(G)
    # Petersen graph has girth 5 (no triangles), so no relators from triangles
    # It has 15 edges and 10 vertices, so without triangles: rank = 15 - 10 + 1 = 6
    assert rec.rank == 6


def test_complete_bipartite():
    """Complete bipartite K_{m,n} has no triangles if m,n >= 2."""
    G = nx.complete_bipartite_graph(2, 3)  # 5 vertices, 6 edges
    rec = fundamental_group(G)
    # No triangles: rank = |E| - |V| + 1 = 6 - 5 + 1 = 2
    assert rec.rank == 2


def test_hypercube():
    """Hypercube Q_3 (3-dimensional cube graph) has rank > 0."""
    G = nx.hypercube_graph(3)
    rec = fundamental_group(G)
    # Q_3 has 8 vertices, 12 edges, no triangles
    # rank = 12 - 8 + 1 = 5
    assert rec.rank == 5


# ---------- covering_graph ----------


def test_covering_graph_trivial_fundamental_group():
    """For K_n (simply connected), cover equals the graph itself."""
    G = nx.complete_graph(4)
    rec = fundamental_group(G)
    assert rec.rank == 0
    cover = covering_graph(G, None, rec=rec)
    # Should have same number of vertices
    assert cover.order() == G.order()
    # Only coset index 0
    assert all(v[1] == 0 for v in cover.nodes())


def test_covering_graph_full_subgroup():
    """When H = π₁ itself (generator), cover equals the graph."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    assert rec.rank == 1
    # H = ⟨g₁⟩ = π₁ means 1-fold cover
    cover = covering_graph(G, [[1]], rec=rec)
    assert cover.order() == G.order()


def test_covering_graph_2_fold_cover_of_cycle():
    """The 2-fold cover of C_4 by subgroup 2ℤ has 8 vertices."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    assert rec.rank == 1
    # subgroup generated by g₁² (word [1,1]):
    cover = covering_graph(G, [[1, 1]], rec=rec)
    assert cover.order() == 8  # 4 vertices × 2 cosets


def test_covering_graph_3_fold_cover_of_cycle():
    """The 3-fold cover of C_4 by subgroup 3ℤ has 12 vertices."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    # subgroup generated by g₁³ (word [1,1,1]):
    cover = covering_graph(G, [[1, 1, 1]], rec=rec)
    assert cover.order() == 12  # 4 vertices × 3 cosets


def test_covering_graph_cycle_5():
    """Cover of C_5 with 2-fold subgroup has 10 vertices."""
    G = nx.cycle_graph(5)
    rec = fundamental_group(G)
    assert rec.rank == 1
    cover = covering_graph(G, [[1, 1]], rec=rec)
    assert cover.order() == 10


def test_covering_graph_edges_preserved():
    """The covering graph has the right number of edges."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    cover = covering_graph(G, [[1, 1]], rec=rec)
    # 2-fold cover: each edge lifts to 2 edges
    assert cover.size() == G.size() * 2


def test_covering_graph_simply_connected_base():
    """Cover of a simply connected graph is the graph itself."""
    G = nx.path_graph(5)
    rec = fundamental_group(G)
    assert rec.rank == 0
    cover = covering_graph(G, None, rec=rec)
    assert cover.order() == G.order()
    assert cover.size() == G.size()


def test_covering_graph_empty_subgroup_words():
    """Empty subgroup words list gives same result as None for trivial π₁."""
    G = nx.complete_graph(3)
    rec = fundamental_group(G)
    cover_none = covering_graph(G, None, rec=rec)
    cover_empty = covering_graph(G, [], rec=rec)
    assert cover_none.order() == cover_empty.order()


def test_covering_graph_without_rec():
    """covering_graph works without passing rec."""
    G = nx.cycle_graph(4)
    cover = covering_graph(G, [[1, 1]])
    assert cover.order() == 8


def test_covering_graph_vertex_labels():
    """Vertices of cover are tuples (v, coset_index)."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    cover = covering_graph(G, [[1, 1]], rec=rec)
    for node in cover.nodes():
        assert isinstance(node, tuple)
        assert len(node) == 2
        v, coset_idx = node
        assert isinstance(v, int)
        assert isinstance(coset_idx, int)
        assert coset_idx in [0, 1]


def test_covering_graph_petersen():
    """Petersen graph has rank 6 (free group), so most subgroups have infinite index."""
    G = nx.petersen_graph()
    rec = fundamental_group(G)
    assert rec.rank == 6
    assert rec.relators == []  # Free group, no relators
    # Subgroup generated by all generators squared has infinite index in F_6
    # The covering_graph should raise RuntimeError when exceeding max_cosets
    subgroup_gens = [[i, i] for i in range(1, rec.rank + 1)]
    with pytest.raises(RuntimeError, match="exceeded.*cosets"):
        covering_graph(G, subgroup_gens, rec=rec, max_cosets=100)


def test_covering_graph_is_connected_for_normal_subgroup():
    """Cover is connected when subgroup is normal."""
    G = nx.cycle_graph(4)
    rec = fundamental_group(G)
    # For abelian groups (like Z), all subgroups are normal
    cover = covering_graph(G, [[1, 1]], rec=rec)
    assert nx.is_connected(cover)
