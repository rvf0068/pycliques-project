"""Zero-degree attachments into a single critical circle."""

from __future__ import annotations

import networkx as nx
import pytest
from pycliques import completely_pared_graph
from pycombtop import (
    facet_morse_certificate,
    homotopy_type_with_verdict,
    integral_morse_certificate,
    symbolic_integral_morse_certificate,
    verify_integral_morse,
)


def _circle_sphere_wedge() -> nx.Graph:
    graph = nx.disjoint_union(nx.cycle_graph(4), nx.octahedral_graph())
    graph = nx.contracted_nodes(graph, 0, 4, self_loops=False)
    return nx.convert_node_labels_to_integers(graph)


def test_one_circle_integral_and_facet_certificates() -> None:
    graph = _circle_sphere_wedge()
    facets = list(nx.find_cliques(graph))
    masks = {sum(1 << v for v in q) for q in facets}
    integral = integral_morse_certificate(masks)
    assert integral is not None
    assert integral.spheres == {1: 1, 2: 1}
    assert verify_integral_morse(masks, integral)
    explicit = facet_morse_certificate(facets)
    assert explicit is not None
    assert explicit.spheres == {1: 1, 2: 1}


@pytest.mark.parametrize("code", [b"I}KGGGB?w", b"IsX___J@o"])
def test_cubic_complements_one_circle(code: bytes) -> None:
    graph = completely_pared_graph(nx.complement(nx.from_graph6_bytes(code)))
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    result = symbolic_integral_morse_certificate(graph)
    assert result is not None
    assert result.critical == {1: 1, 2: 1}
    assert result.wedge_spheres == {1: 1, 2: 1}
    verdict = homotopy_type_with_verdict(graph)
    assert verdict.is_exact and verdict.wedge.spheres == {1: 1, 2: 1}


def test_symbolic_nonzero_degree_not_certified_as_circle_wedge() -> None:
    # A cone on a circle is contractible, not a surviving circle plus sphere.
    graph = nx.wheel_graph(5)
    result = symbolic_integral_morse_certificate(graph)
    assert result is not None
    assert result.wedge_spheres != {1: 1, 2: 1}
