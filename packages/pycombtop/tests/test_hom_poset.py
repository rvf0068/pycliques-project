"""Regression tests for the Hom poset, including reflexive graphs."""

from itertools import combinations, product

import networkx as nx
import pytest
from pycombtop import (
    HomPosetLimit,
    betti_numbers_sc,
    clique_complex,
    complete_poset,
    face_poset,
    graph_homomorphisms,
    hom_graph,
    hom_order_complex,
    hom_poset,
    is_multihomomorphism,
    iter_multihomomorphisms,
    order_complex,
)


def _brute_force(g, h):
    subsets = [frozenset(s) for k in range(1, len(h) + 1) for s in combinations(h, k)]
    result = set()
    for values in product(subsets, repeat=len(g)):
        eta = dict(zip(g, values, strict=True))
        if all(h.has_edge(x, y) for u, v in g.edges() for x in eta[u] for y in eta[v]):
            result.add(frozenset(eta.items()))
    return result


def _looped_graph(n, edges, loops):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    graph.add_edges_from((v, v) for v in loops)
    return graph


@pytest.mark.parametrize(
    "g,h",
    [
        (nx.path_graph(3), nx.complete_graph(3)),
        (nx.empty_graph(2), nx.path_graph(3)),
        (nx.complete_graph(3), nx.complete_graph(2)),
        (nx.empty_graph(0), nx.empty_graph(0)),
        (nx.empty_graph(1), nx.empty_graph(0)),
        (_looped_graph(1, [], [0]), _looped_graph(3, [(0, 1)], [0, 1, 2])),
        (_looped_graph(2, [(0, 1)], [0]), _looped_graph(3, [(0, 1)], [0, 1])),
        (_looped_graph(1, [], [0]), nx.complete_graph(2)),
    ],
)
def test_hom_poset_against_definition(g, h):
    p = hom_poset(g, h)
    expected = _brute_force(g, h)
    assert set(p) == expected == set(iter_multihomomorphisms(g, h))
    for eta in p:
        first = dict(eta)
        assert is_multihomomorphism(g, h, first)
        for tau in p:
            second = dict(tau)
            assert p.leq(eta, tau) == all(first[v] <= second[v] for v in g)
    atoms = {
        frozenset((v, next(iter(image))) for v, image in eta)
        for eta in p.minimal_elements()
    }
    assert atoms == {frozenset(f.items()) for f in graph_homomorphisms(g, h)}
    assert set(hom_graph(g, h)) == atoms


def test_k2_k3_hom_circle_and_graph_bridges():
    g = nx.complete_graph(2)
    h = nx.complete_graph(3)
    p = hom_poset(g, h)
    assert len(p) == 12
    assert len(p.minimal_elements()) == len(p.maximal_elements()) == 6
    assert nx.is_isomorphic(p.comparability_graph(), nx.cycle_graph(12))
    assert nx.is_isomorphic(hom_graph(g, h), nx.cycle_graph(6))
    x = hom_order_complex(g, h)
    assert x == order_complex(p) == clique_complex(p.comparability_graph())
    assert betti_numbers_sc(x) == [0, 1]
    assert complete_poset(h) == face_poset(clique_complex(h))


def test_single_vertex_and_labels():
    g = nx.Graph()
    g.add_node("source")
    h = nx.Graph()
    h.add_nodes_from(["a", 4, ("b",)])
    p = hom_poset(g, h)
    assert len(p) == 7
    assert len(p.maximal_elements()) == 1
    assert hom_order_complex(g, h).dimension() == 2
    assert is_multihomomorphism(g, h, {"source": ["a", 4]})
    assert not is_multihomomorphism(g, h, {"source": []})
    assert not is_multihomomorphism(g, h, {"source": [99]})
    assert not is_multihomomorphism(g, h, {})
    assert not is_multihomomorphism(g, h, {"source": [4], "extra": [4]})


def test_loop_cross_products_not_just_singleton_maps():
    g = _looped_graph(1, [], [0])
    h = _looped_graph(3, [(0, 1)], [0, 1, 2])
    p = hom_poset(g, h)
    assert len(p) == 4
    assert is_multihomomorphism(g, h, {0: {0, 1}})
    assert not is_multihomomorphism(g, h, {0: {0, 2}})
    assert not is_multihomomorphism(g, nx.complete_graph(3), {0: {0}})
    assert graph_homomorphisms(g, nx.complete_graph(3)) == []


def test_budget_and_graph_category():
    g, h = nx.complete_graph(2), nx.complete_graph(3)
    assert len(hom_poset(g, h, max_elements=12)) == 12
    assert len(hom_poset(g, h, max_elements=None)) == 12
    with pytest.raises(HomPosetLimit, match="max_elements"):
        hom_poset(g, h, max_elements=11)
    with pytest.raises(HomPosetLimit):
        hom_poset(g, h, max_elements=0)
    assert len(hom_poset(nx.complete_graph(3), g, max_elements=0)) == 0
    for bound in (-1, True, 2.5):
        with pytest.raises(ValueError, match="max_elements"):
            hom_poset(g, h, max_elements=bound)
    for operation in (hom_poset, hom_order_complex, graph_homomorphisms, hom_graph):
        with pytest.raises(ValueError, match="undirected"):
            operation(nx.DiGraph(), h)
    with pytest.raises(ValueError, match="undirected"):
        is_multihomomorphism(g, nx.DiGraph(), {})
