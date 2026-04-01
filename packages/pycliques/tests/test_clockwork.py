"""Tests for the clockwork module."""

import itertools

import networkx as nx
import pytest
from pycliques.clockwork import (
    ClockworkGraph,
    ClockworkStructure,
    clockwork_graph,
    core,
    crown,
    crown_general,
    has_induced_4cycle,
    is_clique_divergent_clockwork,
    is_complete_graph,
    recognize_as_clockwork_graph,
    recognize_clockwork,
    remove_dominated_vertices,
    segmented_sum,
)

# -------------------------------------------------------------------
# Test graph builders (extracted from demo code)
# -------------------------------------------------------------------


def _build_clockwork_divergent() -> nx.Graph:
    """Build a clique-DIVERGENT clockwork graph (s=4).

    Core: C0={1}, C1={2,3}, C2={4}, C3={5,6}
    Crown: B0={7,8}, B1={9,10}, B2={11,12}, B3={13,14}
    Properties: 2 good segments, 0 covered vertices.
    """
    G = nx.Graph()

    # Core intra-segment edges
    G.add_edge(2, 3)
    G.add_edge(5, 6)

    # Core cross-segment edges
    G.add_edge(1, 2)  # C0→C1
    G.add_edge(3, 4)  # C1→C2
    G.add_edge(4, 5)  # C2→C3
    G.add_edge(6, 1)  # C3→C0

    # Crown intra-segment edges
    for pair in [(7, 8), (9, 10), (11, 12), (13, 14)]:
        G.add_edge(*pair)

    # Crown inter-segment matchings
    G.add_edges_from([(7, 9), (8, 10)])
    G.add_edges_from([(9, 11), (10, 12)])
    G.add_edges_from([(11, 13), (12, 14)])
    G.add_edges_from([(13, 7), (14, 8)])

    # Segmented sum: B_i adj C_{i-1} ∪ C_i
    for b in [7, 8]:
        for c in [5, 6, 1]:
            G.add_edge(b, c)
    for b in [9, 10]:
        for c in [1, 2, 3]:
            G.add_edge(b, c)
    for b in [11, 12]:
        for c in [2, 3, 4]:
            G.add_edge(b, c)
    for b in [13, 14]:
        for c in [4, 5, 6]:
            G.add_edge(b, c)

    return G


def _build_clockwork_bounded() -> nx.Graph:
    """Build a clique-BOUNDED clockwork graph (s=4).

    Core: C0={1,2}, C1={3,4}, C2={5,6}, C3={7,8}
    Crown: B0={9,10}, B1={11,12}, B2={13,14}, B3={15,16}
    Full cross-adjacency: 0 good segments, 4 covered vertices.
    """
    G = nx.Graph()
    s = 4
    C_segs = [{1, 2}, {3, 4}, {5, 6}, {7, 8}]
    B_segs = [{9, 10}, {11, 12}, {13, 14}, {15, 16}]

    C_list = [sorted(seg) for seg in C_segs]
    B_list = [sorted(seg) for seg in B_segs]

    # Core: cliques within each segment
    for seg in C_segs:
        for a, b in itertools.combinations(seg, 2):
            G.add_edge(a, b)

    # Core: full cross-adjacency
    for i in range(s):
        for u in C_list[i]:
            for v in C_list[(i + 1) % s]:
                G.add_edge(u, v)

    # Crown: cliques within each segment
    for bi in B_list:
        G.add_edge(bi[0], bi[1])

    # Crown: perfect matchings
    for i in range(s):
        bi = B_list[i]
        bj = B_list[(i + 1) % s]
        G.add_edge(bi[0], bj[0])
        G.add_edge(bi[1], bj[1])

    # Segmented sum: B_i adj C_{i-1} ∪ C_i
    for i in range(s):
        for b in B_list[i]:
            for c in C_list[(i - 1) % s] + C_list[i]:
                G.add_edge(b, c)

    return G


# -------------------------------------------------------------------
# Graph predicates
# -------------------------------------------------------------------


def test_has_induced_4cycle_in_cycle():
    """C4 itself is an induced 4-cycle."""
    assert has_induced_4cycle(nx.cycle_graph(4), [0, 1, 2, 3])


def test_has_induced_4cycle_not_in_complete():
    """K4 has no induced C4."""
    assert not has_induced_4cycle(nx.complete_graph(4), [0, 1, 2, 3])


def test_has_induced_4cycle_too_few_vertices():
    """Fewer than 4 vertices cannot contain C4."""
    assert not has_induced_4cycle(nx.cycle_graph(5), [0, 1, 2])


def test_is_complete_graph_true():
    """Three vertices of K4 form a clique."""
    assert is_complete_graph(nx.complete_graph(4), [0, 1, 2])


def test_is_complete_graph_false():
    """Three vertices of P4 do not form a clique."""
    assert not is_complete_graph(nx.path_graph(4), [0, 1, 2])


def test_is_complete_graph_single_vertex():
    """A single vertex is trivially a clique."""
    assert is_complete_graph(nx.path_graph(1), [0])


# -------------------------------------------------------------------
# Graph constructors
# -------------------------------------------------------------------


def test_core_basic():
    """Core with segments [1,1,1] has 3 vertices."""
    G, segs = core([1, 1, 1], [[1], [0], [0]])
    assert G.number_of_nodes() == 3
    assert len(segs) == 3
    assert segs == [[0], [1], [2]]


def test_core_segments_partition() -> None:
    """Core segments partition the vertex set."""
    segments = [2, 1, 2]
    neig = [[1, 1], [2], [0, 1]]
    G, segs = core(segments, neig)
    all_verts: set[int] = set()
    for s in segs:
        all_verts.update(s)
    assert all_verts == set(range(sum(segments)))


def test_core_segments_are_cliques():
    """Each core segment is a clique."""
    G, segs = core([2, 2, 2], [[1, 2], [1, 2], [1, 2]])
    for seg in segs:
        for u, v in itertools.combinations(seg, 2):
            assert G.has_edge(u, v)


def test_crown_basic():
    """Crown with 3 segments of size 2 has 6 vertices."""
    G, segs = crown(3, 2, [1, 0])
    assert G.number_of_nodes() == 6
    assert len(segs) == 3


def test_crown_segments_are_cliques():
    """Each crown segment is a clique."""
    G, segs = crown(4, 3, [2, 0, 1])
    for seg in segs:
        for u, v in itertools.combinations(seg, 2):
            assert G.has_edge(u, v)


def test_crown_general_basic():
    """crown_general produces correct vertex count."""
    G, segs = crown_general(2, [[0, 1], [0, 1], [1, 0]])
    assert G.number_of_nodes() == 6
    assert len(segs) == 3


def test_crown_general_matches_crown():
    """crown_general with identity + wrap matches crown."""
    perm = [1, 0]
    G1, _ = crown(3, 2, perm)
    matchings = [[0, 1], [0, 1], perm]
    G2, _ = crown_general(2, matchings)
    assert nx.is_isomorphic(G1, G2)


def test_segmented_sum_vertex_count():
    """Segmented sum has |B| + |C| vertices."""
    b = crown(3, 2, [1, 0])
    c = core([1, 1, 1], [[1], [0], [0]])
    G = segmented_sum(b, c)
    assert G.number_of_nodes() == 6 + 3


def test_clockwork_graph_basic():
    """clockwork_graph returns a connected graph."""
    G = clockwork_graph([1, 1, 1], [[1], [0], [0]], 2, [1, 0])
    assert G.number_of_nodes() == 9
    assert nx.is_connected(G)


# -------------------------------------------------------------------
# Recognition
# -------------------------------------------------------------------


def test_recognize_divergent_clockwork():
    """Hand-built divergent clockwork is recognised."""
    G = _build_clockwork_divergent()
    ok, result = recognize_clockwork(G)
    assert ok
    assert isinstance(result, ClockworkStructure)
    assert result.s == 4
    assert len(result.good_segments) == 2


def test_recognize_bounded_clockwork():
    """Hand-built bounded clockwork is recognised."""
    G = _build_clockwork_bounded()
    ok, result = recognize_clockwork(G)
    assert ok
    assert isinstance(result, ClockworkStructure)
    assert result.s == 4


def test_recognize_cycle_not_clockwork():
    """C6 is not a clockwork graph."""
    ok, _ = recognize_clockwork(nx.cycle_graph(6))
    assert not ok


def test_recognize_complete_not_clockwork():
    """K5 is not a clockwork graph."""
    ok, _ = recognize_clockwork(nx.complete_graph(5))
    assert not ok


def test_recognize_petersen_not_clockwork():
    """The Petersen graph is not clockwork."""
    ok, _ = recognize_clockwork(nx.petersen_graph())
    assert not ok


def test_recognize_empty_graph():
    """An empty graph is not clockwork."""
    ok, reason = recognize_clockwork(nx.Graph())
    assert not ok
    assert isinstance(reason, str)


def test_recognize_disconnected_graph():
    """A disconnected graph is not clockwork."""
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    ok, _ = recognize_clockwork(G)
    assert not ok


def test_recognize_from_clockwork_graph_function():
    """A graph built by clockwork_graph() is recognised."""
    G = clockwork_graph([1, 1, 1], [[1], [0], [0]], 2, [1, 0])
    ok, result = recognize_clockwork(G)
    assert ok
    assert isinstance(result, ClockworkStructure)


def test_recognize_random_graph_not_clockwork():
    """A random G(20, 0.4) graph is not clockwork."""
    G = nx.erdos_renyi_graph(20, 0.4, seed=42)
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
    ok, _ = recognize_clockwork(G)
    assert not ok


# -------------------------------------------------------------------
# Clique divergence
# -------------------------------------------------------------------


def test_is_clique_divergent_true():
    """Divergent clockwork graph is classified as divergent."""
    G = _build_clockwork_divergent()
    div, _ = is_clique_divergent_clockwork(G)
    assert div is True


def test_is_clique_divergent_false():
    """Bounded clockwork graph is classified as bounded."""
    G = _build_clockwork_bounded()
    div, _ = is_clique_divergent_clockwork(G)
    assert div is False


def test_is_clique_divergent_non_clockwork():
    """Non-clockwork graph returns None."""
    div, _ = is_clique_divergent_clockwork(nx.cycle_graph(6))
    assert div is None


# -------------------------------------------------------------------
# ClockworkGraph class
# -------------------------------------------------------------------


def test_clockwork_graph_class_basic():
    """ClockworkGraph construction and basic properties."""
    cg = ClockworkGraph(
        [1, 1, 1],
        [[1], [0], [0]],
        2,
        [[0, 1], [0, 1], [1, 0]],
    )
    assert cg.graph.number_of_nodes() == 9
    assert cg.s == 3


def test_clockwork_graph_repr_roundtrip():
    """eval(repr(cg)) produces an isomorphic graph."""
    cg = ClockworkGraph(
        [1, 1, 1],
        [[1], [0], [0]],
        2,
        [[0, 1], [0, 1], [1, 0]],
    )
    cg2 = eval(repr(cg))  # noqa: S307
    assert cg == cg2


def test_clockwork_graph_str():
    """str() produces a multi-line summary."""
    cg = ClockworkGraph(
        [1, 1, 1],
        [[1], [0], [0]],
        2,
        [[0, 1], [0, 1], [1, 0]],
    )
    s = str(cg)
    assert "ClockworkGraph" in s
    assert "s (segments)" in s


def test_clockwork_graph_eq_not_implemented():
    """Comparing with a non-ClockworkGraph returns NotImpl."""
    cg = ClockworkGraph(
        [1, 1, 1],
        [[1], [0], [0]],
        2,
        [[0, 1], [0, 1], [1, 0]],
    )
    assert cg.__eq__("not a graph") is NotImplemented


def test_clockwork_graph_validation_neig_mismatch():
    """Mismatched core_segments/neig_segments raises ValueError."""
    with pytest.raises(ValueError, match="neig_segments"):
        ClockworkGraph(
            [1, 1],
            [[1], [0], [0]],
            2,
            [[0, 1], [0, 1], [1, 0]],
        )


def test_clockwork_graph_validation_matchings_mismatch():
    """Mismatched core_segments/crown_matchings raises."""
    with pytest.raises(ValueError, match="crown_matchings"):
        ClockworkGraph(
            [1, 1, 1],
            [[1], [0], [0]],
            2,
            [[0, 1], [0, 1]],
        )


def test_clockwork_graph_from_structure():
    """from_structure produces isomorphic graph."""
    G = _build_clockwork_divergent()
    ok, struct = recognize_clockwork(G)
    assert ok
    assert isinstance(struct, ClockworkStructure)
    cg = ClockworkGraph.from_structure(struct)
    assert nx.is_isomorphic(cg.graph, G)


def test_clockwork_graph_from_structure_roundtrip():
    """from_structure -> repr -> eval round-trips."""
    G = _build_clockwork_divergent()
    ok, struct = recognize_clockwork(G)
    assert ok
    assert isinstance(struct, ClockworkStructure)
    cg = ClockworkGraph.from_structure(struct)
    cg2 = eval(repr(cg))  # noqa: S307
    assert nx.is_isomorphic(cg.graph, cg2.graph)


# -------------------------------------------------------------------
# Convenience wrapper
# -------------------------------------------------------------------


def test_recognize_as_clockwork_graph_success():
    """recognize_as_clockwork_graph succeeds on valid graph."""
    G = clockwork_graph([1, 1, 1], [[1], [0], [0]], 2, [1, 0])
    ok, cg = recognize_as_clockwork_graph(G)
    assert ok
    assert isinstance(cg, ClockworkGraph)


def test_recognize_as_clockwork_graph_failure():
    """recognize_as_clockwork_graph fails on non-clockwork."""
    ok, reason = recognize_as_clockwork_graph(nx.cycle_graph(6))
    assert not ok
    assert isinstance(reason, str)


# -------------------------------------------------------------------
# remove_dominated_vertices
# -------------------------------------------------------------------


def test_remove_dominated_vertices_path():
    """Removing dominated vertices from P4 leaves 1 vertex."""
    H = remove_dominated_vertices(nx.path_graph(4))
    assert H.number_of_nodes() == 1


def test_remove_dominated_vertices_cycle():
    """No dominated vertices in C5."""
    G = nx.cycle_graph(5)
    H = remove_dominated_vertices(G)
    assert H.number_of_nodes() == 5


def test_remove_dominated_vertices_complete():
    """K4: all dominated, leaves one vertex."""
    H = remove_dominated_vertices(nx.complete_graph(4))
    assert H.number_of_nodes() == 1


# -------------------------------------------------------------------
# ClockworkStructure repr
# -------------------------------------------------------------------


def test_clockwork_structure_repr():
    """ClockworkStructure repr includes key info."""
    G = _build_clockwork_divergent()
    ok, struct = recognize_clockwork(G)
    assert ok
    assert isinstance(struct, ClockworkStructure)
    r = repr(struct)
    assert "ClockworkStructure" in r
    assert "Good segments" in r
