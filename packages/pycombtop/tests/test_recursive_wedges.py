"""Regression tests for replayable recursive cofibers and signed Morse chains."""

from dataclasses import replace

import networkx as nx
import pytest
from pycombtop import Theorem, homotopy_type_with_verdict
from pycombtop.integral_morse import (
    IntegralMorseCertificate,
    bipartite_shore_facets,
    integral_morse_certificate,
    verify_integral_morse,
)
from pycombtop.recursive_wedges import (
    ImageCarrier,
    RecursiveWedgeCertificate,
    WedgeStep,
    _check_carrier,
    recursive_wedge_certificate,
    verify_recursive_wedge,
)


def test_discrete_link_carried_by_a_path_in_a_square():
    # Three disjoint length-two paths between vertices 0 and 2 form K_{2,3}.
    # Deleting 4 leaves a square. Its path 0-1-2 contracts the two-point link.
    h = nx.cycle_graph(4)
    h.add_edges_from([(4, 0), (4, 2)])
    carrier = ImageCarrier((), (), (1,), ((0, 1), (2, 1)))
    split = WedgeStep("split", (15, 5), vertex=4, carrier=carrier)
    steps = {5: WedgeStep("leaf"), 15: WedgeStep("leaf"), 31: split}
    cert = RecursiveWedgeCertificate(tuple(h), steps, {1: 2})
    assert verify_recursive_wedge(h, cert)
    bad = replace(split, carrier=replace(carrier, extra=()))
    assert not verify_recursive_wedge(h, replace(cert, steps={**steps, 31: bad}))
    assert not verify_recursive_wedge(h, replace(cert, spheres={1: 1}))
    assert not verify_recursive_wedge(h, replace(cert, steps={31: split}))
    found = recursive_wedge_certificate(h)
    assert found and found.spheres == cert.spheres
    assert recursive_wedge_certificate(h, max_states=1) is None
    assert recursive_wedge_certificate(h, max_vertices=len(h) - 1) is None


def test_essential_attaching_map_fails_closed():
    # The hub cones off the square. Pendant vertex 4 is a contractible
    # singleton outside the hub's link and cannot carry that square.
    h = nx.cycle_graph(4)
    h.add_edge(0, 4)
    h.add_edges_from((5, v) for v in range(4))
    step = WedgeStep("split", (31, 15), vertex=5)
    steps = {15: WedgeStep("leaf"), 31: WedgeStep("leaf"), 63: step}
    cert = RecursiveWedgeCertificate(tuple(h), steps, {1: 1, 2: 1})
    assert not verify_recursive_wedge(h, cert)
    forged = ImageCarrier((), (), (4,), ())
    assert not verify_recursive_wedge(
        h, replace(cert, steps={**steps, 63: replace(step, carrier=forged)})
    )


def test_carrier_tracks_link_images_moved_outside_the_link():
    deletion = nx.path_graph(4)
    link = deletion.subgraph([0, 1, 3])
    # Folding 0 onto 1 in the source leaves {1, 3}. After the two
    # deletion folds the image is {2, 3}, whose edge contracts onto 3.
    carrier = ImageCarrier(((0, 1),), ((0, 1), (1, 2)), (), ((2, 3),))
    assert _check_carrier(deletion, link, carrier)
    assert not _check_carrier(
        deletion, link, replace(carrier, extra=(3,), contraction=())
    )
    # Neither the link nor the deletion allows the proposed fold of 1 onto 3.
    assert not _check_carrier(deletion, link, replace(carrier, link_folds=((1, 3),)))
    assert not _check_carrier(
        deletion, link, replace(carrier, deletion_folds=((1, 3),))
    )


def test_recursive_search_builds_and_replays_an_image_carrier():
    graph = nx.circulant_graph(12, [1, 3, 4])
    proof = recursive_wedge_certificate(graph)
    assert proof and proof.spheres == {1: 1, 2: 4}
    assert verify_recursive_wedge(graph, proof)
    mask, step = next(
        (m, s) for m, s in proof.steps.items() if s.carrier and s.carrier.extra
    )
    shortened = replace(step, carrier=replace(step.carrier, extra=()))
    assert not verify_recursive_wedge(
        graph, replace(proof, steps={**proof.steps, mask: shortened})
    )


def test_recursive_search_is_inconclusive_on_a_flag_torus():
    graph = nx.grid_2d_graph(4, 4, periodic=True)
    # Triangulate every square with the same diagonal, including across the seams.
    graph.add_edges_from(
        ((i, j), ((i + 1) % 4, (j + 1) % 4)) for i in range(4) for j in range(4)
    )
    assert {len(q) for q in nx.find_cliques(graph)} == {3}
    assert recursive_wedge_certificate(graph) is None


def test_replay_checks_fold_survivors_and_join_factors():
    graph = nx.octahedral_graph()
    graph.add_edge(0, 6)
    proof = recursive_wedge_certificate(graph)
    assert proof and proof.spheres == {2: 1} and verify_recursive_wedge(graph, proof)
    root = (1 << len(graph)) - 1
    fold = proof.steps[root]
    assert fold.rule == "fold" and fold.moves == ((6, 0),)
    # Skipping the pendant fold leaves vertex 6 in the live graph.
    bad = replace(fold, moves=())
    assert not verify_recursive_wedge(
        graph, replace(proof, steps={**proof.steps, root: bad})
    )
    (core,) = fold.children
    join = proof.steps[core]
    assert join.rule == "join" and len(join.children) == 3
    # Each pair of antipodal vertices contributes one S^0 factor.
    bad = replace(join, children=join.children[:-1])
    assert not verify_recursive_wedge(
        graph, replace(proof, steps={**proof.steps, core: bad})
    )


@pytest.mark.parametrize(
    "graph", [nx.Graph(), nx.DiGraph([(0, 1)]), nx.Graph([(0, 0)])]
)
def test_certificate_methods_reject_invalid_flag_graphs(graph):
    assert recursive_wedge_certificate(graph) is None
    assert not verify_recursive_wedge(graph, RecursiveWedgeCertificate((), {}, {}))
    assert bipartite_shore_facets(graph) is None


def test_discrete_link_and_arbitrary_labels():
    h = nx.Graph([("a", 1), (1, ("v",)), (("v",), "b"), ("b", "a")])
    cert = recursive_wedge_certificate(h)
    assert cert and cert.spheres == {1: 1} and verify_recursive_wedge(h, cert)
    assert bipartite_shore_facets(nx.empty_graph(3)) is None
    assert bipartite_shore_facets(nx.empty_graph(2)) is None


def test_integral_morse_replay_rejects_bad_counts_pairs_and_budget():
    # A triangulated sphere admits one critical 0-cell and one critical 2-cell.
    facets = {7, 11, 13, 14}
    cert = integral_morse_certificate(facets)
    assert cert and cert.spheres == {2: 1} and verify_integral_morse(facets, cert)
    assert not verify_integral_morse(facets, replace(cert, spheres={2: 2}))
    assert not verify_integral_morse(facets, replace(cert, removals=cert.removals[1:]))
    assert not verify_integral_morse(
        facets, replace(cert, removals=((1, 7),) + cert.removals)
    )
    assert not verify_integral_morse(facets, cert, max_faces=13)


@pytest.mark.parametrize(
    "move",
    [
        (0, -1),
        (-1, 1),
        (0, 0),
        (0, 1 << 500),
        (1 << 500, 1),
        (0, 2),
        (4, 9),
        (True, 1),
        (0, True),
        (False, 1),
        (0.0, 1),
        (None, 1),
    ],
)
def test_invalid_morse_masks_rejected_before_coface_scan(move, monkeypatch):
    from pycombtop import integral_morse

    def unexpected_scan(mask):
        pytest.fail(f"Invalid removal reached the coface scan with mask {mask}")

    monkeypatch.setattr(integral_morse, "_bit_list", unexpected_scan)
    # Mask 2 is a hole in the vertex set. Masks 4 and 9 are live faces
    # whose numeric order alone would allow the invalid pair (4, 9).
    cert = IntegralMorseCertificate((move,), {})
    assert not verify_integral_morse({4, 9}, cert)


def test_signed_boundary_detects_suspended_projective_plane():
    # Its CW model has one cell each in dimensions 0, 2 and 3; the degree
    # of the 3-cell attachment is +/-2, invisible over F_2.
    triangles = ["123", "124", "135", "146", "156", "236", "245", "256", "345", "346"]
    masks = [sum(1 << (int(v) - 1) for v in tri) for tri in triangles]
    suspension = {face | (1 << pole) for face in masks for pole in (6, 7)}
    assert integral_morse_certificate(suspension) is None


def test_integral_replay_requires_a_maximal_critical_face_and_complete_moves():
    facets = {7}
    # Vertex 0 is still contained in two edges of the filled triangle.
    assert not verify_integral_morse(facets, IntegralMorseCertificate(((0, 1),), {}))
    assert not verify_integral_morse(facets, IntegralMorseCertificate(((7,),), {}))
    assert integral_morse_certificate(facets, max_faces=6) is None
    assert integral_morse_certificate({0}) is None


@pytest.mark.parametrize("d", [2, 3])
def test_mixed_degree_morse_on_two_simplex_boundaries(d):
    # The boundaries of a (d+1)- and a (d+2)-simplex meet at vertex 0.
    left = set(range(d + 2))
    right = {0, *range(d + 2, 2 * d + 4)}
    facets = {
        sum(1 << v for v in simplex - {missing})
        for simplex in (left, right)
        for missing in simplex
    }
    cert = integral_morse_certificate(facets)
    assert cert and cert.spheres == {d: 1, d + 1: 1}
    assert verify_integral_morse(facets, cert)
    assert not verify_integral_morse(facets, replace(cert, spheres={d: 2}))


def test_shore_of_even_cycle_complement():
    graph = nx.complement(nx.cycle_graph(8))
    shore = bipartite_shore_facets(graph)
    assert shore and len(shore) == 4 and all(f.bit_count() == 2 for f in shore)
    cert = integral_morse_certificate(shore)
    assert cert and cert.spheres == {1: 1}
    verdict = homotopy_type_with_verdict(graph)
    assert verdict.is_exact and verdict.wedge.spheres == {2: 1}
    assert homotopy_type_with_verdict(graph, max_vertices=7).is_limited


def test_classifier_uses_recursive_proof_with_a_small_face_budget():
    verdict = homotopy_type_with_verdict(nx.cycle_graph(25), max_simplices=10)
    assert verdict.is_exact and verdict.reason == Theorem.RECURSIVE_VERTEX_STAR
    assert verdict.wedge.spheres == {1: 1}


def test_classifier_uses_integral_morse_when_recursive_search_is_inconclusive(
    monkeypatch,
):
    import pycombtop.homotopy_type as ht

    monkeypatch.setattr(ht, "recursive_wedge_certificate", lambda graph: None)
    circle = nx.cycle_graph(25)
    # The cycle has 51 faces including the empty face.
    direct = homotopy_type_with_verdict(circle, max_simplices=51)
    assert direct.is_exact and direct.reason == Theorem.INTEGRAL_MORSE
    assert direct.wedge.spheres == {1: 1}
    assert homotopy_type_with_verdict(circle, max_simplices=50).is_limited

    # The independence complex of C_(3k+2) is S^k. Its much smaller shore
    # fits the face budget and contributes the extra suspension dimension.
    graph = nx.complement(nx.cycle_graph(26))
    shore = homotopy_type_with_verdict(graph, max_simplices=8192)
    assert (
        shore.is_exact
        and shore.reason == Theorem.BIPARTITE_SHORE + Theorem.INTEGRAL_MORSE
    )
    assert shore.wedge.spheres == {8: 1}
