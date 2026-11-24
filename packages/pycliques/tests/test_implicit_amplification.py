"""Replay amplification and distinguish finite growth from divergence."""

from dataclasses import replace
from itertools import product

import networkx as nx
import pytest
from pycliques import (
    ImplicitCliqueGraph,
    RetractionSearchStatus,
    amplified_common_neighbor,
    certify_octahedral_amplification,
    completely_pared_graph,
    find_complete_transversal,
    octahedron,
    verify_octahedral_amplification,
)

PAIRS = ((0, 1), (2, 3), (4, 5))


def test_certified_amplification_and_tampering() -> None:
    host = octahedron(3)
    cert = certify_octahedral_amplification(host, PAIRS)
    assert cert.dimension == 4
    assert len(cert.antipodal_pairs) == 4
    assert verify_octahedral_amplification(host, cert)
    assert cert.extensions == cert.common_neighborhoods
    for all_extensions in (False, True):
        assert (
            amplified_common_neighbor(host, cert, all_extensions=all_extensions).status
            is RetractionSearchStatus.EXHAUSTED
        )
    assert not verify_octahedral_amplification(
        host, replace(cert, extensions=cert.extensions[:-1])
    )
    assert not verify_octahedral_amplification(
        host, replace(cert, extensions=cert.extensions[::-1])
    )
    assert not verify_octahedral_amplification(
        host, replace(cert, common_neighborhoods=cert.common_neighborhoods[:-1])
    )
    with pytest.raises(ValueError, match="invalid amplification"):
        amplified_common_neighbor(host, replace(cert, extensions=()))
    with pytest.raises(ValueError, match="max_facets"):
        certify_octahedral_amplification(host, PAIRS, max_facets=7)
    with pytest.raises(ValueError, match="positive integer"):
        certify_octahedral_amplification(host, PAIRS, max_facets=True)


def _intersection_graph(graph: nx.Graph) -> tuple[nx.Graph, dict]:
    implicit = ImplicitCliqueGraph.from_graph(graph)
    assert implicit is not None
    explicit = implicit.to_networkx()
    assert explicit is not None
    return explicit, {implicit.clique(v): v for v in implicit}


def test_two_successful_amplifications_do_not_guarantee_continuation() -> None:
    graph = nx.from_graph6_bytes(b"J]~vJZZn}~?")
    first = certify_octahedral_amplification(graph, PAIRS)
    assert verify_octahedral_amplification(graph, first)
    assert (
        amplified_common_neighbor(graph, first, all_extensions=True).status
        is RetractionSearchStatus.EXHAUSTED
    )
    host, labels = _intersection_graph(graph)
    assert len(host) == 21
    pairs = [(labels[a], labels[b]) for a, b in first.antipodal_pairs]
    second = certify_octahedral_amplification(host, pairs)
    assert second.dimension == 8
    assert verify_octahedral_amplification(host, second)
    result = amplified_common_neighbor(host, second)
    assert result.status is RetractionSearchStatus.FOUND
    assert result.transversal is not None
    assert all(result.transversal & q for q in second.extensions)


def test_every_extension_choice_of_counterexample_is_obstructed() -> None:
    graph = nx.from_graph6_bytes(b"J]~vJZZn}~?")
    first = certify_octahedral_amplification(graph, PAIRS)
    host, labels = _intersection_graph(graph)
    second_cliques = [frozenset(q) for q in nx.find_cliques(host)]
    choices = [
        [
            q
            for q in labels
            if all(pair[(i >> j) & 1] in q for j, pair in enumerate(PAIRS))
        ]
        for i in range(8)
    ]
    count = 0
    for extensions in product(*choices):
        count += 1
        cert = replace(first, extensions=extensions)
        assert verify_octahedral_amplification(graph, cert)
        pairs = [(labels[a], labels[b]) for a, b in cert.antipodal_pairs]
        forced = []
        for i in range(16):
            seed = {pair[(i >> j) & 1] for j, pair in enumerate(pairs)}
            possible = [q for q in second_cliques if seed <= q]
            forced.append(frozenset.intersection(*possible))
        # A complete set meets the intersection of all possible
        # extensions of each second-stage transversal.
        result = find_complete_transversal(host, forced)
        assert result.status is RetractionSearchStatus.FOUND
    assert count == 8


@pytest.mark.slow
def test_cubic_65_two_amplifications_without_constructing_k4() -> None:
    # Reproduce the stable mask ordering of the research certificate.
    graph = completely_pared_graph(nx.complement(nx.from_graph6_bytes(b"KsX_o_C?_B_M")))
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    for order in (13, 24, 140):
        host, _ = _intersection_graph(graph)
        graph = nx.convert_node_labels_to_integers(
            completely_pared_graph(host), ordering="sorted"
        )
        assert len(graph) == order
    pairs = [
        (128, 26),
        (1, 139),
        (17, 121),
        (23, 109),
        (28, 61),
        (29, 126),
        (35, 50),
        (42, 84),
        (47, 112),
        (49, 76),
        (54, 111),
        (58, 110),
        (77, 103),
        (82, 94),
    ]
    cert = certify_octahedral_amplification(graph, pairs, max_facets=16_384)
    assert cert.dimension == 8192
    assert verify_octahedral_amplification(graph, cert)
    for all_extensions in (False, True):
        result = amplified_common_neighbor(
            graph, cert, all_extensions=all_extensions, max_states=100_000
        )
        assert result.status is RetractionSearchStatus.EXHAUSTED
