"""Exact nullhomotopic spherical-link attachment regressions."""

import networkx as nx
import pycombtop.homotopy_type as homotopy
from pycombtop.attachment_certificates import (
    ConeImageCertificate,
    bounded_cone_image_certificate,
    verify_cone_image_certificate,
)
from pycombtop.s_collapses import (
    complete_s_collapse_with_certificate,
    verify_s_collapse_move,
)
from pycombtop.wedge_certificates import (
    FoldedWedgeCertificate,
    certified_flag_wedge,
    certified_flag_wedge_after_s_collapses,
    verify_folded_wedge_certificate,
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


def test_connected_c5_link_into_a_three_sphere():
    # The star-deletion cofiber formula also works with a connected
    # noncontractible link: there are then zero extra S^1 summands.
    graph = _join_graphs(nx.cycle_graph(5), nx.cycle_graph(4))
    apex = len(graph)
    graph.add_edges_from((apex, v) for v in range(5))
    local = homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, apex)
    assert local is not None and local.is_exact
    assert local.wedge.spheres == {2: 1, 3: 1}
    verdict = homotopy.homotopy_type_with_verdict(graph, max_simplices=32)
    assert verdict.is_exact
    assert verdict.wedge.spheres == {2: 1, 3: 1}


def test_wedge_recognizer_fail_closed_on_unrecognized_complexes():
    """Neither a missing type nor a search-budget refusal is a certificate."""
    assert certified_flag_wedge(nx.Graph()) is None
    assert certified_flag_wedge(nx.cycle_graph(5), max_calls=0) is None

    # The icosahedral graph is not classified by any of the deliberately
    # narrow exact rules. In particular, it has triangles, no cutvertex,
    # and a connected complement.
    unknown = nx.icosahedral_graph()
    assert certified_flag_wedge(unknown) is None

    # A join with a contractible factor is contractible, but a join with
    # an unknown factor must remain UNKNOWN (not just infer its Betti type).
    cone = _join_graphs(nx.cycle_graph(5), nx.complete_graph(1))
    assert certified_flag_wedge(cone) == {}
    unidentified_join = _join_graphs(unknown, nx.cycle_graph(4))
    assert certified_flag_wedge(unidentified_join) is None

    # Force an articulation-point recursion with a genuinely uncertified
    # piece. The recognizer must reject the entire decomposition.
    with_pendant = unknown.copy()
    with_pendant.add_edge(0, len(unknown))
    assert set(nx.articulation_points(with_pendant)) == {0}
    assert certified_flag_wedge(with_pendant) is None


def test_spherical_link_rejects_empty_or_disconnected_deletions():
    """Nullhomotopy requires a correctly certified connected deletion."""
    isolated = nx.empty_graph(3)
    assert homotopy._try_nullhomotopic_spherical_link_at_vertex(isolated, 0) is None
    assert homotopy._try_nullhomotopic_spherical_link(isolated) is None

    # In the wheel, the hub has a C5 link, which is a certified S1.
    # Adding a separate isolated vertex makes the deletion disconnected.
    disconnected = nx.disjoint_union(nx.wheel_graph(6), nx.complete_graph(1))
    assert certified_flag_wedge(disconnected.subgraph(disconnected.neighbors(0))) == {
        1: 1
    }
    assert homotopy._try_nullhomotopic_spherical_link_at_vertex(disconnected, 0) is None


def test_spherical_link_rejects_uncertified_deletion_type():
    """A known S1 link does not determine an unknown attaching target."""
    graph = nx.icosahedral_graph()
    circle_vertices = list(graph.neighbors(0))
    # An icosahedron vertex link is an induced C5. A new apex attached
    # to these five vertices has a certified S1 link, but its deletion
    # remains the icosahedral graph, which this recognizer cannot classify.
    apex = len(graph)
    graph.add_edges_from((apex, vertex) for vertex in circle_vertices)
    assert certified_flag_wedge(graph.subgraph(circle_vertices)) == {1: 1}
    assert homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, apex) is None


def _triangle_ear_on_c5() -> nx.Graph:
    """A flag S1 with a triangular ear, not recognized directly."""
    graph = nx.cycle_graph(5)
    graph.add_edges_from([(5, 0), (5, 1)])
    return graph


def test_link_s_collapse_finds_nonoctahedral_sphere():
    link = _triangle_ear_on_c5()
    assert certified_flag_wedge(link) is None
    certificate = certified_flag_wedge_after_s_collapses(link)
    assert certificate is not None
    assert certificate.spheres == {1: 1}
    assert len(certificate.moves) == 1
    assert certificate.moves[0][0] == 5
    assert verify_folded_wedge_certificate(link, certificate)

    # Neither the claimed homotopy type nor a fabricated move is trusted.
    assert not verify_folded_wedge_certificate(
        link, FoldedWedgeCertificate({2: 1}, certificate.moves)
    )
    assert not verify_folded_wedge_certificate(
        link, FoldedWedgeCertificate({1: 1}, ((0, ()),))
    )


def test_link_fold_budgets_reject_without_guessing():
    link = _triangle_ear_on_c5()
    assert certified_flag_wedge_after_s_collapses(link, max_vertices=5) is None
    assert certified_flag_wedge_after_s_collapses(link, max_moves=0) is None
    assert certified_flag_wedge_after_s_collapses(link, max_checks=1) is None
    # A directly recognized cycle needs no reduction budget.
    certificate = certified_flag_wedge_after_s_collapses(
        nx.cycle_graph(5), max_checks=0
    )
    assert certificate is not None and certificate.moves == ()
    assert certificate.spheres == {1: 1}


def test_s_collapsed_link_extends_nullhomotopic_attachment():
    link = _triangle_ear_on_c5()
    graph, vertex = _link_attachment_fixture(link)
    neighborhood = graph.subgraph(graph.neighbors(vertex))
    components = list(nx.connected_components(neighborhood))
    assert sorted(len(c) for c in components) == [1, 6]
    assert certified_flag_wedge(neighborhood.subgraph(components[0])) is None or (
        certified_flag_wedge(neighborhood.subgraph(components[1])) is None
    )
    local = homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, vertex)
    assert local is not None and local.is_exact
    assert local.wedge.spheres == {1: 1, 2: 1, 3: 1}
    # The bounded global heuristic need not select this vertex: other
    # admissible candidates can consume its two-deletion search budget.
    # The local vertex certificate above remains a complete exact proof.


def test_bounded_link_collapse_cannot_bypass_dimension_guard():
    link = _triangle_ear_on_c5()
    graph = link.copy()
    graph.add_edge(6, 0)
    graph.add_edges_from((7, v) for v in range(6))
    graph.add_edge(7, 6)
    # The folded link is S1 but the deletion has type S1, so the
    # degree-one attaching map need not be nullhomotopic.
    assert homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, 7) is None


def test_phase2_10223998_null_attachment_via_cone_image():
    # Another order-10 Phase-2 TIMEOUT core, 22 vertices, 189 edges.
    graph = nx.from_graph6_bytes(
        r"U~h\z}nV|~n~j}V|v|z}~jn|\~|v~zn~~i}nt^Vw".encode("ascii")
    )
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (22, 189)
    link = graph.subgraph(list(graph.neighbors(4))).copy()
    folded = certified_flag_wedge_after_s_collapses(link)
    assert folded is not None
    assert folded.spheres == {6: 1}
    assert [v for v, _local in folded.moves] == [0, 2, 20]
    assert verify_folded_wedge_certificate(link, folded)

    deletion = graph.copy()
    deletion.remove_node(4)
    proof = bounded_cone_image_certificate(deletion, list(link))
    assert proof is not None
    assert [v for v, _dominator in proof.moves] == list(range(5, 18))
    assert proof.spheres == {3: 1}
    assert proof.image_vertices == (0, 2, 18, 19, 20)
    assert proof.apex == 2
    assert verify_cone_image_certificate(deletion, list(link), proof)

    result = homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, 4)
    assert result is not None and result.is_exact
    assert result.wedge.spheres == {3: 1, 7: 1}
    assert result.reason == homotopy.Theorem.NULLHOMOTOPIC_CONE_IMAGE

    # New structural certificate executes before simplex enumeration.
    verdict = homotopy.homotopy_type_with_verdict(
        graph, max_vertices=22, max_simplices=50
    )
    assert verdict.is_exact and verdict.wedge.spheres == {3: 1, 7: 1}
    assert homotopy.Theorem.NULLHOMOTOPIC_CONE_IMAGE in verdict.reason


def test_cone_image_proof_is_replayable_and_budgeted():
    graph = nx.from_graph6_bytes(
        r"U~h\z}nV|~n~j}V|v|z}~jn|\~|v~zn~~i}nt^Vw".encode("ascii")
    )
    link_vertices = list(graph.neighbors(4))
    deletion = graph.copy()
    deletion.remove_node(4)
    proof = bounded_cone_image_certificate(deletion, link_vertices)
    assert proof is not None

    # False cone apex, false homotopy type and invalid domination move
    # must all be detected by a replay, even when the stored graph6 is valid.
    assert not verify_cone_image_certificate(
        deletion,
        link_vertices,
        ConeImageCertificate(proof.moves, proof.spheres, proof.image_vertices, 0),
    )
    assert not verify_cone_image_certificate(
        deletion,
        link_vertices,
        ConeImageCertificate(proof.moves, {2: 1}, proof.image_vertices, proof.apex),
    )
    assert not verify_cone_image_certificate(
        deletion,
        link_vertices,
        ConeImageCertificate(
            ((0, 19),) + proof.moves, proof.spheres, proof.image_vertices, proof.apex
        ),
    )
    assert (
        bounded_cone_image_certificate(deletion, link_vertices, max_vertices=20) is None
    )
    assert (
        bounded_cone_image_certificate(deletion, link_vertices, max_checks=25) is None
    )
    assert bounded_cone_image_certificate(deletion, link_vertices, max_moves=3) is None


def test_forged_s_collapse_move_shapes_are_rejected():
    link = _triangle_ear_on_c5()
    for forged_moves in (
        ((5,),),
        (([], ()),),
        ((5, None),),
        ((5, ([],)),),
    ):
        forged = FoldedWedgeCertificate({1: 1}, forged_moves)
        assert verify_folded_wedge_certificate(link, forged) is False


def test_prebudget_search_reuses_link_certificates(monkeypatch):
    graph, _ = _link_attachment_fixture(_triangle_ear_on_c5())
    original = homotopy.certified_flag_wedge_after_s_collapses
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(homotopy, "certified_flag_wedge_after_s_collapses", counted)
    homotopy._try_nullhomotopic_spherical_link(graph)
    # Only six fallback searches are ever allowed; selected candidates
    # must replay their certificates instead of spending another search.
    assert 1 <= calls <= 6


def test_forged_cone_retraction_move_shapes_are_rejected():
    graph = nx.from_graph6_bytes(
        r"U~h\z}nV|~n~j}V|v|z}~jn|\~|v~zn~~i}nt^Vw".encode("ascii")
    )
    link = list(graph.neighbors(4))
    deletion = graph.copy()
    deletion.remove_node(4)
    proof = bounded_cone_image_certificate(deletion, link)
    assert proof is not None
    for forged_moves in (
        ((5,),) + proof.moves,
        (([], 19),) + proof.moves,
        ((5, []),) + proof.moves,
    ):
        forged = ConeImageCertificate(
            forged_moves, proof.spheres, proof.image_vertices, proof.apex
        )
        assert verify_cone_image_certificate(deletion, link, forged) is False
