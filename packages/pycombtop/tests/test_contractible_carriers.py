"""Exact, bounded contractible-image carriers strictly beyond cone images."""

import networkx as nx
import pycombtop.homotopy_type as homotopy
from pycombtop.attachment_certificates import (
    ConeImageCertificate,
    ContractibleCarrierCertificate,
    bounded_cone_image_certificate,
    bounded_null_image_certificate,
    verify_contractible_image_certificate,
)
from pycombtop.contractible_carriers import (
    bounded_carrier_collapse,
    bounded_contractible_carrier,
    verify_contractible_carrier,
)


def _cycle_to_path_deletion() -> tuple[nx.Graph, tuple[int, ...]]:
    """A C6 link folds onto a P4 carrier inside a deletion of type S1.

    Build the target as C5 with a pendant path 0-5-6-7, and six extra
    cycle vertices 8,...,13. The six clone folds
      8->0, 9->5, 10->6, 11->7, 12->6, 13->5
    map the C6 link onto all four vertices of the path P4. This image
    has NO universal vertex, but its clique complex is contractible.

    Clones are inserted first to make the bounded greedy fold order
    deterministic, matching the recorded ordinary-fold certificate.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(8, 14))
    graph.add_nodes_from(range(8))
    graph.add_edges_from(nx.cycle_graph(5).edges())
    graph.add_edges_from([(0, 5), (5, 6), (6, 7)])
    targets = [0, 5, 6, 7, 6, 5]
    for i, vertex in enumerate(targets):
        for x in [vertex, *graph.neighbors(vertex)]:
            if x < 8:
                graph.add_edge(i + 8, x)
    for i in range(6):
        graph.add_edge(8 + i, 8 + ((i + 1) % 6))
    link = tuple(range(8, 14))
    assert nx.is_isomorphic(graph.subgraph(link), nx.cycle_graph(6))
    return graph, link


def test_path_carrier_is_contractible_but_not_a_cone():
    path = nx.path_graph(4)
    assert not any(all(u == v or path.has_edge(u, v) for u in path) for v in path)
    proof = bounded_carrier_collapse(path)
    assert proof is not None
    moves, survivor = proof
    assert len(moves) == 3
    assert verify_contractible_carrier(path, moves, survivor)
    assert not verify_contractible_carrier(path, moves, "wrong")
    assert not verify_contractible_carrier(path, (([], ()),), survivor)
    assert not verify_contractible_carrier(path, ((0,),), survivor)
    assert bounded_carrier_collapse(path, max_vertices=3) is None
    assert bounded_carrier_collapse(path, max_checks=0) is None
    assert bounded_carrier_collapse(path, max_moves=2) is None


def test_a_carrier_can_include_a_vertex_outside_the_image():
    # The induced image is C5 and is NOT contractible. A center vertex
    # outside the image supplies a contractible wheel as a carrier.
    wheel = nx.wheel_graph(6)
    cycle = tuple(range(1, 6))
    assert bounded_contractible_carrier(wheel, cycle, max_extra_vertices=0) is None
    proof = bounded_contractible_carrier(wheel, cycle)
    assert proof is not None
    carrier, moves, survivor = proof
    assert set(carrier) == set(wheel)
    assert verify_contractible_carrier(wheel, moves, survivor)
    assert bounded_contractible_carrier(wheel, cycle, max_candidates=1) is None
    assert bounded_contractible_carrier(wheel, cycle, max_carrier_vertices=5) is None


def test_exact_same_dimensional_cofiber_uses_noncone_carrier(monkeypatch):
    deletion, link = _cycle_to_path_deletion()
    assert (
        bounded_cone_image_certificate(deletion, link, max_checks=128, max_moves=6)
        is None
    )
    certificate = bounded_null_image_certificate(
        deletion, link, max_checks=128, max_moves=6, max_extra_vertices=0
    )
    assert isinstance(certificate, ContractibleCarrierCertificate)
    assert certificate.moves == (
        (8, 0),
        (9, 5),
        (10, 6),
        (11, 7),
        (12, 6),
        (13, 5),
    )
    assert certificate.spheres == {1: 1}
    assert set(certificate.image_vertices) == {0, 5, 6, 7}
    assert set(certificate.carrier_vertices) == {0, 5, 6, 7}
    assert verify_contractible_image_certificate(deletion, link, certificate)

    # Attach an apex to exactly the six-cycle. The link and deletion
    # both have type S1: the connectivity/dimension-gap argument FAILS,
    # but the contractible carrier proves that the attachment is null.
    graph = deletion.copy()
    apex = 14
    graph.add_edges_from((apex, x) for x in link)
    original_search = homotopy.bounded_null_image_certificate

    def bounded_six_folds(target, images, **kwargs):
        return original_search(target, images, **(kwargs | {"max_moves": 6}))

    monkeypatch.setattr(homotopy, "bounded_null_image_certificate", bounded_six_folds)
    verdict = homotopy._try_nullhomotopic_spherical_link_at_vertex(graph, apex)
    assert verdict is not None and verdict.is_exact
    assert verdict.reason == homotopy.Theorem.NULLHOMOTOPIC_CONTRACTIBLE_CARRIER
    assert verdict.wedge.spheres == {1: 1, 2: 1}

    # The ordinary cone path still returns the original certificate
    # class and remains compatible with all earlier regressions.
    original = bounded_null_image_certificate(deletion, link)
    assert isinstance(original, ConeImageCertificate)


def test_contractible_carrier_verifier_rejects_forged_proofs():
    deletion, link = _cycle_to_path_deletion()
    valid = bounded_null_image_certificate(
        deletion, link, max_moves=6, max_extra_vertices=0
    )
    assert isinstance(valid, ContractibleCarrierCertificate)

    def with_change(**updates):
        parts = dict(
            moves=valid.moves,
            spheres=valid.spheres,
            image_vertices=valid.image_vertices,
            carrier_vertices=valid.carrier_vertices,
            carrier_moves=valid.carrier_moves,
            survivor=valid.survivor,
        )
        return ContractibleCarrierCertificate(**(parts | updates))

    forged = [
        with_change(spheres={2: 1}),
        with_change(image_vertices=(0,)),
        with_change(carrier_vertices=(0, 5, 6)),
        with_change(survivor=999),
        with_change(carrier_moves=(([], ()),)),
        with_change(carrier_moves=((0,),)),
        with_change(moves=(([], 0),)),
    ]
    assert all(
        verify_contractible_image_certificate(deletion, link, valid)
        and not verify_contractible_image_certificate(deletion, link, bad)
        for bad in forged
    )

    assert (
        bounded_null_image_certificate(
            deletion, link, max_moves=6, max_carrier_candidates=0
        )
        is None
    )
    assert (
        bounded_null_image_certificate(
            deletion, link, max_moves=6, max_carrier_vertices=3
        )
        is None
    )
    assert (
        bounded_null_image_certificate(deletion, link, max_moves=6, max_checks=0)
        is None
    )


def test_carrier_can_need_two_extra_vertices_to_connect_image():
    path = nx.path_graph(4)
    image = (0, 3)
    assert bounded_contractible_carrier(path, image, max_extra_vertices=1) is None
    assert bounded_contractible_carrier(path, image, max_candidates=3) is None
    result = bounded_contractible_carrier(path, image, max_candidates=4)
    assert result is not None
    vertices, moves, survivor = result
    assert set(vertices) == set(path)
    assert verify_contractible_carrier(path, moves, survivor)


def test_carrier_budget_and_malformed_input_fail_closed():
    path = nx.path_graph(4)
    assert bounded_carrier_collapse(nx.Graph()) is None
    assert bounded_carrier_collapse(nx.empty_graph(2)) is None
    assert bounded_carrier_collapse(nx.cycle_graph(5)) is None
    assert bounded_carrier_collapse(path, max_moves=-1) is None
    assert bounded_contractible_carrier(path, ()) is None
    assert bounded_contractible_carrier(path, ([],)) is None
    assert bounded_contractible_carrier(path, (100,)) is None
    assert bounded_contractible_carrier(path, (0,), max_extra_vertices=-1) is None
    assert bounded_contractible_carrier(path, (0,), max_candidates=0) is None
    assert bounded_contractible_carrier(path, (0,), max_carrier_vertices=0) is None
    assert bounded_null_image_certificate(path, (0,), max_checks=-1) is None
    assert bounded_null_image_certificate(path, (0,), max_moves=-1) is None
    assert bounded_null_image_certificate(path, ([],)) is None
