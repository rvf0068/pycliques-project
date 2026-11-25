"""Mixed homology, adaptive bounds and deterministic symbolic retries."""

import base64

import networkx as nx
import pytest
from pycombtop import (
    HomotopyVerdict,
    Theorem,
    WedgeOfSpheres,
    recursive_betti_upper_bounds,
    symbolic_integral_morse_certificate,
    verify_connected_link_order,
)
from pycombtop.bounded_betti import bounded_reduced_betti_mod2

# Fixed labelled first pared iterate of cubic complement 138.
CORE = (
    "c35+fn5+fn5+fn5+fn5+fn5+fn5+P357Y355eU54d258Sk5+e21Ofkh0fnxNaH5jfV5+"
    "Zmpifn53fn5+fW5+fn5yfn5+fn5Ofn5+bV5+fn5tfn5+fm5Ofn5+enp+fndGfn5+b2Nm"
    "fn5+SFZQfn59ak5Efn59WmhYfn5+Un5kcH59Tn54TW5+dHJ+aHROfm5Oe2Zyfn1Yfnt8"
    "W15+aH5+fnd7XkZ+fn59blZkcn5+fnJ4dVh+fn5+b1ldZn5+fn5veF1qfn5+fmRxW1p+"
    "fn5+Z1ZkeH5+fn56VExTfn5+"
)
ORDER = (
    36,
    49,
    16,
    5,
    37,
    3,
    50,
    23,
    1,
    29,
    47,
    11,
    20,
    8,
    0,
    48,
    30,
    21,
    12,
    27,
    15,
    18,
    41,
    51,
    32,
    26,
    44,
    35,
    42,
    31,
    24,
    9,
    10,
    39,
    34,
    17,
    7,
    43,
    28,
    4,
    46,
    6,
    33,
    38,
    14,
    40,
    13,
    19,
    22,
    25,
    2,
    45,
)


def test_relative_hurewicz_mixed_critical_dimensions(monkeypatch):
    import pycombtop.symbolic_integral_morse as sim

    graph = nx.from_graph6_bytes(base64.b64decode(CORE))
    assert (len(graph), graph.number_of_edges()) == (52, 1068)
    result = symbolic_integral_morse_certificate(graph, vertex_order=ORDER)
    assert result is not None
    assert result.critical == {4: 1, 5: 29, 6: 32}
    assert result.betti == {5: 3, 6: 7}
    assert result.torsion == {}
    assert result.wedge_spheres == result.betti
    assert result.connectivity_order is not None
    assert verify_connected_link_order(graph, result.connectivity_order)
    # Homology by itself must not certify the wedge in this branch.
    monkeypatch.setattr(sim, "connected_link_order", lambda graph: None)
    unsupported = symbolic_integral_morse_certificate(graph, vertex_order=ORDER)
    assert unsupported is not None
    assert unsupported.betti == result.betti
    assert unsupported.wedge_spheres is None


def test_symbolic_classifier_tries_distinct_degree_orders(monkeypatch):
    import pycombtop.homotopy_type as ht

    graph = nx.path_graph(5)
    calls = []
    expected = HomotopyVerdict(WedgeOfSpheres({}), Theorem.SYMBOLIC_DONG_MATCHING)

    def attempt(graph, order=None):
        calls.append(order)
        return expected if order is not None else None

    monkeypatch.setattr(ht, "_try_symbolic_dong_order", attempt)
    assert ht._try_symbolic_dong(graph) == expected
    assert calls == [None, [1, 2, 3, 0, 4]]


def test_adaptive_bounds_against_exact_small_graphs():
    for graph in nx.graph_atlas_g():
        if len(graph) > 6:
            break
        result = recursive_betti_upper_bounds(graph, adaptive_pivots=2)
        assert not result.is_limited
        exact = bounded_reduced_betti_mod2(graph, max_degree=5)
        assert exact is not None
        assert all(rank >= exact.get(d, 0) for d, rank in result.upper_bounds.items())
        chi = -1 + sum(
            (-1) ** (len(face) - 1) for face in nx.enumerate_all_cliques(graph)
        )
        assert result.reduced_euler_characteristic == chi


def test_adaptive_limits_and_join():
    graph = nx.complete_multipartite_graph(2, 2, 2, 2)
    result = recursive_betti_upper_bounds(graph, adaptive_pivots=2)
    assert result.upper_bounds[3] == 1
    assert sum(result.upper_bounds.values()) == 1
    assert result.reduced_euler_characteristic == -1
    limited = recursive_betti_upper_bounds(graph, adaptive_pivots=2, max_states=1)
    assert limited.is_limited
    assert limited.upper_bounds == {}
    assert limited.reduced_euler_characteristic is None
    for bad in (-1, True):
        with pytest.raises(ValueError):
            recursive_betti_upper_bounds(graph, adaptive_pivots=bad)
    with pytest.raises(ValueError):
        recursive_betti_upper_bounds(graph, max_degree=2, adaptive_pivots=2)
