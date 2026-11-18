"""Exact symbolic Morse boundaries and certified conicity bounds."""

import base64

import networkx as nx
import pytest
from pycombtop import (
    Theorem,
    find_weighted_conicity_certificate,
    homotopy_type_with_verdict,
    symbolic_integral_morse_certificate,
    verify_weighted_conicity,
)

# Fixed graph6 of K^3(Susp(C7)); deterministic vertex order matters.
C7_K3_B64 = (
    "d35+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn"
    "5+fn5+fn5+fn5+fn5vY3Fefj9CSn59XmU/Xn1EZUB+fkVyP15+b0tvfn59P0B+fn53P0h+"
    "fj9Ofn5Cfj9Ofk5ifmNGfn1AfnhQfn4/Xn5tTn5vQn57X359dU5+eHh+bj9efnx4fmU/Xn"
    "57P35Lfn5+fj9Oek5yfn5+eF5lP15+fn5sdVdCfn5+P25aTnp+fn4/bVJOfn5+fnddQGZ+"
    "fn5+d0ZhWH5+fn5+UHNSTn5+fn5+Rj9Lfn5+fn59bT9Qfn5+fn59fT9Rfn5+fn57"
)


def test_symbolic_integral_morse_c7_k3_and_classifier():
    graph = nx.from_graph6_bytes(base64.b64decode(C7_K3_B64))
    assert (len(graph), graph.number_of_edges()) == (56, 1190)
    proof = symbolic_integral_morse_certificate(graph)
    assert proof is not None
    assert proof.total_faces == 546_132_471
    assert proof.critical == {3: 1, 4: 2}
    assert proof.boundaries == {4: ((0, 0),)}
    assert proof.betti == {3: 1, 4: 2}
    assert proof.torsion == {}
    assert proof.wedge_spheres == {3: 1, 4: 2}
    assert proof.gradient_states > 0
    verdict = homotopy_type_with_verdict(
        graph, max_vertices=64, max_simplices=25_000
    )
    assert verdict.is_exact and not verdict.is_limited
    assert verdict.wedge.spheres == {3: 1, 4: 2}
    assert verdict.reason == Theorem.SYMBOLIC_INTEGRAL_MORSE


def test_nonzero_integer_morse_boundary_is_not_misread_as_wedge():
    # A nonzero incidence between a critical vertex and critical edge.
    graph = nx.from_graph6_bytes(b"E`MW")
    result = symbolic_integral_morse_certificate(
        graph, vertex_order=[0, 2, 5, 3, 1, 4]
    )
    assert result is not None
    assert result.critical == {0: 1, 1: 1}
    assert result.boundaries == {1: ((1,),)}
    assert result.betti == {}
    assert result.torsion == {}
    assert result.wedge_spheres is None


@pytest.mark.parametrize(
    "graph,expected",
    [
        (nx.path_graph(5), {}),
        (nx.cycle_graph(5), {1: 1}),
        (nx.octahedral_graph(), {2: 1}),
        (nx.empty_graph(3), {0: 2}),
    ],
)
def test_integral_symbolic_standard_cases(graph, expected):
    result = symbolic_integral_morse_certificate(graph)
    assert result is not None
    assert result.betti == expected
    assert result.wedge_spheres == expected


def test_symbolic_integral_resource_limits():
    graph = nx.from_graph6_bytes(base64.b64decode(C7_K3_B64))
    assert symbolic_integral_morse_certificate(graph, max_critical=2) is None
    assert symbolic_integral_morse_certificate(graph, max_gradient_states=1) is None
    assert symbolic_integral_morse_certificate(graph, max_nodes=10) is None
    with pytest.raises(ValueError, match="vertex_order"):
        symbolic_integral_morse_certificate(
            nx.cycle_graph(5), vertex_order=[0, 1]
        )


def test_weighted_conicity_integer_witness_and_solver():
    graph = nx.from_graph6_bytes(b"F^~~w")
    uniform = verify_weighted_conicity(graph, dict.fromkeys(graph, 1))
    assert uniform.conicity == 3
    certificate = find_weighted_conicity_certificate(graph, time_limit=5)
    assert certificate is not None
    assert certificate.conicity == 5
    assert certificate.connectivity == 1
    assert certificate.total_weight > 5 * certificate.max_closed_non_neighbor_weight
    recovered = verify_weighted_conicity(
        graph, dict(zip(certificate.vertices, certificate.weights))
    )
    assert recovered == certificate


def test_weighted_conicity_validation_and_fallback():
    graph = nx.complete_graph(9)
    certificate = verify_weighted_conicity(graph, dict.fromkeys(graph, 1))
    assert (certificate.conicity, certificate.connectivity) == (8, 3)
    assert find_weighted_conicity_certificate(graph) == certificate
    assert find_weighted_conicity_certificate(graph, max_vertices=8) is None
    with pytest.raises(ValueError, match="exactly"):
        verify_weighted_conicity(graph, {0: 1})
    with pytest.raises(ValueError, match="nonnegative"):
        verify_weighted_conicity(graph, dict.fromkeys(graph, -1))
    with pytest.raises(ValueError, match="positive"):
        verify_weighted_conicity(graph, dict.fromkeys(graph, 0))
