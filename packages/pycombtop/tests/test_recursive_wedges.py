"""Regression tests for replayable recursive cofibers and signed Morse chains."""

from dataclasses import replace

import networkx as nx
import pytest
from pycombtop import homotopy_type_with_verdict
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
