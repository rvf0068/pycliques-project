"""Regression tests for replayable recursive cofibers and signed Morse chains."""

from dataclasses import replace
from itertools import combinations

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


def test_source_folded_carrier_on_residual_census_core():
    # McKay 11661360. Rebuild K(G) using the census ordering of cliques.
    graph = nx.from_graph6_bytes(b"IFzfvr~~o")
    cliques = sorted(tuple(sorted(q)) for q in nx.find_cliques(graph))
    h = nx.empty_graph(len(cliques))
    h.add_edges_from(
        (i, j)
        for i, j in combinations(h, 2)
        if set(cliques[i]).intersection(cliques[j])
    )
    cert = recursive_wedge_certificate(h)
    assert cert and cert.spheres == {3: 4, 5: 6, 7: 16}
    root = (1 << len(h)) - 1
    split = cert.steps[root]
    assert split.carrier and split.carrier.extra
    bad = replace(split, carrier=replace(split.carrier, extra=()))
    assert not verify_recursive_wedge(h, replace(cert, steps={**cert.steps, root: bad}))
    assert not verify_recursive_wedge(h, replace(cert, spheres={5: 26}))
    assert not verify_recursive_wedge(h, replace(cert, steps={root: split}))
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


def test_mixed_degree_morse_on_residual_index_11716425():
    g = nx.from_graph6_bytes(b"IUzrv~}~_")
    cliques = sorted(tuple(sorted(q)) for q in nx.find_cliques(g))
    h = nx.empty_graph(len(cliques))
    h.add_edges_from(
        (i, j)
        for i, j in combinations(h, 2)
        if set(cliques[i]).intersection(cliques[j])
    )
    verdict = homotopy_type_with_verdict(h, max_vertices=64, max_simplices=25000)
    assert verdict.is_exact and verdict.wedge.spheres == {3: 1, 4: 3}
    assert verdict.reason == Theorem.INTEGRAL_MORSE
    assert homotopy_type_with_verdict(h, max_vertices=20).is_limited
