"""Cross-check symbolic Dong matching against explicit simplex matching."""

import base64
import gc
import random
import weakref
from collections import Counter
from itertools import combinations

import networkx as nx
import pycombtop.symbolic_dong as symbolic_dong
import pytest
from pycombtop import (
    Theorem,
    clique_complex,
    dong_matching,
    homotopy_type_with_verdict,
    symbolic_dong_certificate,
)

# Reproducible graph6 of K^3(Susp(C6)), encoded to avoid literal backslashes.
C6_K3_G6_B64 = (
    "d35+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn"
    "5+fn5+fn5+fn5+fn54e0tefmQ/fn5+SWBufn4/Rn5+fmZzY35+eFxAfn5+TnhIXn58Xmtg"
    "fn55bnV2fj9Gfn5+fl9Ofn54fn5bfnt2fn5hfn59Qn5+bl57Y25+e15+fkdefnxefnpHXn"
    "57Qn5+fU5+fkJ+fnJifn53YHxefn5+fmFOVn56fn5+fXV9fV5+fn57TmF9fn5+fmlGcH5+"
    "fn5+en1yfV5+fn5+XE5Cdn5+fn5+c3tCfn5+fn5+e2lEfn5+fn5+eGdGfn5+fn57"
)


def c6_third_iterate():
    return nx.from_graph6_bytes(base64.b64decode(C6_K3_G6_B64))


def explicit_critical_masks(graph, order):
    """Independent face-set implementation of sequential matching."""
    vertices = list(order)
    faces = set()
    for k in range(len(vertices) + 1):
        for face in combinations(range(len(vertices)), k):
            if all(
                graph.has_edge(vertices[i], vertices[j])
                for i, j in combinations(face, 2)
            ):
                faces.add(sum(1 << i for i in face))
    unmatched = set(faces)
    for i in range(len(vertices)):
        pairs = []
        for face in unmatched:
            if face & (1 << i):
                continue
            upper = face | (1 << i)
            if upper in unmatched:
                pairs.extend((face, upper))
        unmatched.difference_update(pairs)
    return unmatched


def test_symbolic_matches_independent_enumeration_and_dong():
    rng = random.Random(84217)
    for n in range(1, 9):
        for _ in range(30):
            graph = nx.gnp_random_graph(n, rng.random(), seed=rng.randrange(1 << 30))
            order = list(graph)
            rng.shuffle(order)
            certificate = symbolic_dong_certificate(graph, vertex_order=order)
            assert certificate is not None
            expected = explicit_critical_masks(graph, order)
            assert set(certificate.critical_masks) == expected
            assert certificate.critical == dict(
                Counter(mask.bit_count() - 1 for mask in expected)
            )
            ordinary = dong_matching(
                clique_complex(graph), order_function=lambda _vertices: order
            )
            indices = {v: i for i, v in enumerate(order)}
            direct_masks = {sum(1 << indices[v] for v in face) for face in ordinary}
            assert direct_masks == expected


def test_known_examples_and_reduced_zero_cells():
    assert symbolic_dong_certificate(nx.path_graph(5)).critical == {}
    assert symbolic_dong_certificate(nx.cycle_graph(5)).critical == {1: 1}
    assert symbolic_dong_certificate(nx.octahedral_graph()).critical == {2: 1}
    assert symbolic_dong_certificate(nx.empty_graph(3)).critical == {0: 2}
    assert symbolic_dong_certificate(nx.Graph()).critical == {-1: 1}


def test_resource_caps_are_inconclusive_and_order_is_validated():
    graph = nx.cycle_graph(6)
    assert symbolic_dong_certificate(graph, max_nodes=1) is None
    assert symbolic_dong_certificate(graph, max_vertices=5) is None
    with pytest.raises(ValueError, match="vertex_order"):
        symbolic_dong_certificate(graph, vertex_order=[0, 1, 2])
    with pytest.raises(ValueError, match="max_nodes"):
        symbolic_dong_certificate(graph, max_nodes=0)


def test_c6_third_iterate_certificate():
    graph = c6_third_iterate()
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (56, 1300)
    certificate = symbolic_dong_certificate(graph)
    assert certificate is not None
    assert certificate.total_faces == 657_325_163
    assert certificate.critical == {7: 11}
    assert certificate.critical_masks is not None
    assert len(certificate.critical_masks) == 11
    assert all(mask.bit_count() == 8 for mask in certificate.critical_masks)
    assert symbolic_dong_certificate(graph, max_nodes=100) is None


def test_classifier_uses_symbolic_matching_before_simplex_budget():
    graph = c6_third_iterate()
    result = homotopy_type_with_verdict(graph, max_vertices=64, max_simplices=25_000)
    assert result.is_exact and not result.is_limited
    assert result.wedge.spheres == {7: 11}
    assert result.reason == Theorem.SYMBOLIC_DONG_MATCHING

def test_diagrams_released_after_success_and_node_limit(monkeypatch):
    """Neither a successful calculation nor a refusal may retain a ZDD."""
    references = []
    original = symbolic_dong._ZDD

    class TrackingZDD(original):
        def __init__(self, n, node_limit):
            super().__init__(n, node_limit)
            references.append(weakref.ref(self))

    monkeypatch.setattr(symbolic_dong, "_ZDD", TrackingZDD)
    graph = nx.cycle_graph(6)
    for _ in range(5):
        assert symbolic_dong_certificate(graph) is not None
        assert symbolic_dong_certificate(graph, max_nodes=1) is None
        gc.collect()
        assert all(reference() is None for reference in references)
        assert original.face_size_counts.cache_info().currsize == 0
        assert original.without_vertices.cache_info().currsize == 0
        assert original.pivot.cache_info().currsize == 0
        assert original.difference.cache_info().currsize == 0
        assert original.contains_empty.cache_info().currsize == 0
        assert original.remove_empty.cache_info().currsize == 0


def test_final_face_count_cache_cleared_on_late_exception(monkeypatch):
    """An exception after counting critical faces also releases the ZDD."""
    references = []
    original = symbolic_dong._ZDD

    class FailingZDD(original):
        def __init__(self, n, node_limit):
            super().__init__(n, node_limit)
            references.append(weakref.ref(self))

        def masks(self, root, prefix=0):
            raise symbolic_dong._DiagramLimit

    monkeypatch.setattr(symbolic_dong, "_ZDD", FailingZDD)
    assert symbolic_dong_certificate(nx.cycle_graph(5)) is None
    gc.collect()
    assert all(reference() is None for reference in references)
    assert original.face_size_counts.cache_info().currsize == 0


