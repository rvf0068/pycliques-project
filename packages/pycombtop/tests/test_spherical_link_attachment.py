"""Exact nullhomotopic spherical-link attachment regressions."""

import networkx as nx
import pycombtop.homotopy_type as homotopy
from pycombtop.wedge_certificates import certified_flag_wedge
from pycombtop.s_collapses import (
    complete_s_collapse_with_certificate,
    verify_s_collapse_move,
)


def test_phase2_11670395_octahedral_link_deletion_certificate():
    # Phase 1's 15-vertex pared clique graph for order-10 census index
    # 11670395. Do not replace graph6 with a relabelled isomorphic graph.
    graph = nx.from_graph6_bytes(b"N~`Xz~_tW\\jnk~M~m~w")
    assert graph.number_of_nodes() == 15
    assert graph.number_of_edges() == 73

    link = graph.subgraph(graph.neighbors(0))
    components = [
        link.subgraph(nodes).copy() for nodes in nx.connected_components(link)
    ]
    assert sorted(len(c) for c in components) == [1, 6]
    sphere_link = next(c for c in components if len(c) == 6)
    assert homotopy._octahedral_sphere_dimension(sphere_link) == 2
    assert any(
        nx.is_isomorphic(c, nx.complete_graph(1)) for c in components if len(c) == 1
    )

    deletion = graph.copy()
    deletion.remove_node(0)
    replayed_core, moves = complete_s_collapse_with_certificate(deletion)
    assert [vertex for vertex, _local in moves] == [1, 2, 3, 8, 9, 12]
    replay = deletion.copy()
    for vertex, local in moves:
        assert verify_s_collapse_move(
            replay,
            {"kind": "vertex", "vertex": vertex, "dismantling": local},
        )
    assert set(replay) == set(replayed_core)
    assert set(replay.edges) == set(replayed_core.edges)
    deletion_core = homotopy._simplify_graph_ht(deletion)
    assert deletion_core.number_of_nodes() == 8
    assert homotopy._octahedral_sphere_dimension(deletion_core) == 3

    local = homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, 0)
    assert local is not None
    assert local.is_exact and not local.is_limited
    assert local.wedge.spheres == {1: 1, 3: 2}
    assert local.reason == homotopy.Theorem.NULLHOMOTOPIC_SPHERICAL_LINK

    # The new certificate executes before even the 50-simplex budget.
    # The original flag complex has 522 nonempty simplices.
    assert homotopy._graph_exceeds_simplex_budget(graph, 50)
    verdict = homotopy.homotopy_type_with_verdict(
        graph, max_vertices=15, max_simplices=50
    )
    assert verdict.is_exact and verdict.wedge.spheres == {1: 1, 3: 2}
    assert homotopy.Theorem.NULLHOMOTOPIC_SPHERICAL_LINK in verdict.reason


def test_dimension_inequality_blocks_essential_link_map():
    # Start with O3 (S^2). Add a pendant path 0--6--7 to the deletion
    # (which s-collapses back to O3), then a vertex 8 adjacent to the
    # O3 vertices and 7, but not to 6.
    # The link at 8 is O3 disjoint-union a point; its S^2 component
    # maps into a deletion of type S^2, so dimensional connectivity
    # does NOT justify a nullhomotopy.
    graph = nx.octahedral_graph()
    graph.add_edges_from([(0, 6), (6, 7)])
    graph.add_edges_from((8, x) for x in range(6))
    graph.add_edge(8, 7)
    assert (
        homotopy._octahedral_sphere_dimension(
            homotopy._simplify_graph_ht(graph.subgraph(range(8)).copy())
        )
        == 2
    )
    assert homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, 8) is None


def test_vertex_limit_overrides_structural_certificate():
    graph = nx.from_graph6_bytes(b"N~`Xz~_tW\\jnk~M~m~w")
    result = homotopy.homotopy_type_with_verdict(
        graph, max_vertices=14, max_simplices=50
    )
    assert result.is_limited and not result.is_exact


def test_sphere_recognizer_rejects_partial_matchings():
    graph = nx.octahedral_graph()
    assert homotopy._octahedral_sphere_dimension(graph) == 2
    graph.add_edge(0, 5)  # This is a missing antipodal edge of O3.
    assert homotopy._octahedral_sphere_dimension(graph) is None
    assert homotopy._octahedral_sphere_dimension(nx.cycle_graph(5)) is None


def test_prebudget_recognizer_never_scans_many_large_deletions(monkeypatch):
    count = 0
    original = homotopy._simplify_graph_ht

    def tracked(graph):
        nonlocal count
        count += 1
        return original(graph)

    monkeypatch.setattr(homotopy, "_simplify_graph_ht", tracked)
    big = nx.disjoint_union_all([nx.cycle_graph(5)] * 6)
    assert big.number_of_nodes() == 30
    assert homotopy._try_nullhomotopic_spherical_link(big) is None
    assert count == 0

    graph = nx.octahedral_graph()
    graph.add_edges_from([(0, 6), (6, 7)])
    graph.add_edges_from((8, x) for x in range(6))
    graph.add_edge(8, 7)
    homotopy._try_nullhomotopic_spherical_link(graph)
    assert count <= 2


def _join_graphs(left: nx.Graph, right: nx.Graph) -> nx.Graph:
    joined = nx.disjoint_union(left, right)
    m, n = len(left), len(right)
    joined.add_edges_from((i, j) for i in range(m) for j in range(m, m + n))
    return joined


def _link_attachment_fixture(link: nx.Graph) -> tuple[nx.Graph, int]:
    # In B = Delta(link * C4), every link factor S1 contributes
    # an S3 after taking the join with C4. Attaching a pendant
    # vertex to a C4 vertex preserves the type of the deletion.
    # The new apex v sees exactly link disjoint-union that pendant.
    g = _join_graphs(link, nx.cycle_graph(4))
    pendant, apex = len(g), len(g) + 1
    g.add_edge(pendant, len(link))
    g.add_edges_from((apex, u) for u in range(len(link)))
    g.add_edge(apex, pendant)
    return g, apex


def test_bounded_wedge_recognizers_beyond_octahedra():
    c5 = nx.cycle_graph(5)
    assert certified_flag_wedge(c5) == {1: 1}
    assert certified_flag_wedge(nx.path_graph(6)) == {}
    assert certified_flag_wedge(nx.octahedral_graph()) == {2: 1}
    assert certified_flag_wedge(nx.empty_graph(3)) == {0: 2}

    two_cycles = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))
    bouquet = nx.contracted_nodes(two_cycles, 0, 4, self_loops=False)
    bouquet = nx.convert_node_labels_to_integers(bouquet)
    assert certified_flag_wedge(bouquet) == {1: 2}
    assert certified_flag_wedge(two_cycles) is None
    joined = _join_graphs(bouquet, nx.cycle_graph(4))
    assert certified_flag_wedge(joined) == {3: 2}
    assert certified_flag_wedge(joined, max_calls=1) is None


def test_nonoctahedral_link_c5_into_join_of_cycles():
    graph, vertex = _link_attachment_fixture(nx.cycle_graph(5))
    deletion = graph.copy()
    deletion.remove_node(vertex)
    assert certified_flag_wedge(homotopy._simplify_graph_ht(deletion)) == {3: 1}
    result = homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, vertex)
    assert result is not None and result.is_exact
    assert result.wedge.spheres == {1: 1, 2: 1, 3: 1}
    limited = homotopy.homotopy_type_with_verdict(graph, max_simplices=32)
    assert limited.is_exact and limited.wedge.spheres == {1: 1, 2: 1, 3: 1}


def test_link_bouquet_of_two_circles_into_wedge_of_three_spheres():
    two_cycles = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))
    bouquet = nx.convert_node_labels_to_integers(
        nx.contracted_nodes(two_cycles, 0, 4, self_loops=False)
    )
    graph, vertex = _link_attachment_fixture(bouquet)
    result = homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, vertex)
    assert result is not None and result.is_exact
    assert result.wedge.spheres == {1: 1, 2: 2, 3: 2}


def test_same_dimension_link_must_not_be_assumed_null():
    # The C5 link maps into the C5 deletion by the identity,
    # which is NOT nullhomotopic. The pendant is an isolated
    # link component and does not make that attachment harmless.
    graph = nx.cycle_graph(5)
    graph.add_edge(5, 0)
    graph.add_edges_from((6, v) for v in range(5))
    graph.add_edge(6, 5)
    assert homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, 6) is None


def test_contractible_deletion_allows_link_circles_without_dimension_gap():
    graph = _join_graphs(nx.cycle_graph(5), nx.complete_graph(1))
    graph.add_edge(6, 5)
    graph.add_edges_from((7, v) for v in range(5))
    graph.add_edge(7, 6)
    result = homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, 7)
    assert result is not None and result.is_exact
    assert result.wedge.spheres == {1: 1, 2: 1}
