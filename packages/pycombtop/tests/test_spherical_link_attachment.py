"""Exact nullhomotopic spherical-link attachment regressions."""

import networkx as nx
import pycombtop.homotopy_type as homotopy


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
        nx.is_isomorphic(c, nx.complete_graph(1))
        for c in components if len(c) == 1
    )

    deletion = graph.copy()
    deletion.remove_node(0)
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
    assert homotopy._octahedral_sphere_dimension(
        homotopy._simplify_graph_ht(graph.subgraph(range(8)).copy())
    ) == 2
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
    graph.add_edge(0, 1)
    assert homotopy._octahedral_sphere_dimension(graph) is None
    assert homotopy._octahedral_sphere_dimension(nx.cycle_graph(5)) is None
