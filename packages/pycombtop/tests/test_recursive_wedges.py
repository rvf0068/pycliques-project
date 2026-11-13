"""Regression tests for replayable recursive cofibers and signed Morse chains."""

from dataclasses import replace
from itertools import combinations

import networkx as nx
from pycombtop import Theorem, homotopy_type_with_verdict
from pycombtop.integral_morse import (
    bipartite_shore_facets,
    integral_morse_certificate,
    verify_integral_morse,
)
from pycombtop.recursive_wedges import (
    ImageCarrier,
    RecursiveWedgeCertificate,
    WedgeStep,
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
    # Coning the whole square fills its essential loop. A forged null proof
    # would turn this contractible wheel into S^1 wedge S^2.
    h = nx.cycle_graph(4)
    h.add_edges_from((4, v) for v in range(4))
    step = WedgeStep("split", (15, 15), vertex=4)
    cert = RecursiveWedgeCertificate(
        tuple(h), {15: WedgeStep("leaf"), 31: step}, {1: 1, 2: 1}
    )
    assert not verify_recursive_wedge(h, cert)
    forged = ImageCarrier((), (), (), ())
    assert not verify_recursive_wedge(
        h,
        replace(cert, steps={15: WedgeStep("leaf"), 31: replace(step, carrier=forged)}),
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
