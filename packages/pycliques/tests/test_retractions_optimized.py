"""Exact regression tests for special octahedra and first-witness retractions."""
# ruff: noqa: I001
from __future__ import annotations

# fmt: off

import itertools
import random

import networkx as nx
import pycliques.retractions as r
import pycliques.small as small
import pytest


def original_special(graph: nx.Graph) -> bool:
    """Independent reference version of the previous nonedge-pair algorithm."""
    nonedges = list(nx.complement(graph).edges())
    auxiliary = nx.Graph()
    auxiliary.add_nodes_from(nonedges)
    for e, f in itertools.combinations(nonedges, 2):
        if set(e).isdisjoint(f) and all(
            graph.has_edge(u, v) for u in e for v in f
        ):
            auxiliary.add_edge(e, f)
    for pairs in nx.find_cliques(auxiliary):
        if len(pairs) < 3:
            continue
        vertices = {v for pair in pairs for v in pair}
        for clique in nx.find_cliques(graph.subgraph(vertices)):
            if r._is_maximal_clique(graph, clique):
                return True
    return False


def octahedron(d: int) -> nx.Graph:
    g = nx.complete_graph(2 * d)
    g.remove_edges_from((2 * i, 2 * i + 1) for i in range(d))
    return g


def check_retraction(host, target, result):
    if result is False:
        return
    projection, inclusion = result
    assert set(projection) == set(host)
    assert set(inclusion) == set(target)
    assert len(set(inclusion.values())) == len(inclusion)
    assert all(projection[inclusion[v]] == v for v in target)
    assert all(
        projection[a] == projection[b]
        or target.has_edge(projection[a], projection[b])
        for a, b in host.edges()
    )
    for x, y in itertools.combinations(target, 2):
        assert host.has_edge(inclusion[x], inclusion[y]) == target.has_edge(x, y)


@pytest.mark.parametrize("graph", [
    nx.cycle_graph(5),
    nx.complete_graph(8),
    octahedron(3),
    octahedron(4),
    nx.disjoint_union(octahedron(3), nx.empty_graph(1)),
    nx.complete_multipartite_graph(2, 2, 2),
])
def test_special_octahedron_examples(graph):
    found = r.special_octahedra_dimension(graph)
    assert (found is not None) == original_special(graph)


def test_all_atlas_graphs_through_seven_vertices():
    for graph in nx.graph_atlas_g():
        found = r.special_octahedra_dimension(graph)
        assert (found is not None) == original_special(graph)


def test_special_octahedron_random_and_relabelled():
    rng = random.Random(20261104)
    for n, density, repetitions in [(10, .25, 12), (10, .6, 12), (20, .4, 8)]:
        for _ in range(repetitions):
            g = nx.gnp_random_graph(n, density, seed=rng.randrange(2**32))
            labels = {v: f"node-{v}" for v in g}
            h = nx.relabel_nodes(g, labels)
            expected = original_special(g)
            assert (r.special_octahedra_dimension(h) is not None) == expected


def test_first_witness_exhaustive_small_graphs():
    targets = (nx.path_graph(2), nx.cycle_graph(4))
    for graph in nx.graph_atlas_g():
        if len(graph) > 5:
            continue
        for target in targets:
            # Keep the original generator's complete enumeration behavior,
            # but compare its first witness with the optimized decision API.
            old = next(r.retraction(graph, target), False)
            new = r.retracts(graph, target)
            assert (old is not False) == (new is not False)
            check_retraction(graph, target, new)


def test_retracts_does_not_list_automorphisms(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("First-witness search must not enumerate automorphisms")
    monkeypatch.setattr(r, "automorphisms", forbidden)
    assert r.retracts(nx.complete_graph(8), nx.cycle_graph(5)) is False
    witness = r.retracts(nx.path_graph(3), nx.path_graph(2))
    check_retraction(nx.path_graph(3), nx.path_graph(2), witness)


def test_pared_iterate_retractions_are_early_and_only_at_zero_and_one(monkeypatch):
    originals = nx.cycle_graph(7)
    iterate1 = nx.cycle_graph(8)
    class FakeSequence:
        def __getitem__(self, index):
            if index == 0:
                return originals
            if index == 1:
                return iterate1
            raise AssertionError("Retraction checked a later iterate")

    calls = []
    def fake_retracts(graph, target):
        calls.append(graph)
        return False
    monkeypatch.setattr(small, "retracts", fake_retracts)
    monkeypatch.setattr(small, "_test_eventually_special_octahedra",
                        lambda seq, tries: calls.append("special") or None)
    tests = small._default_classifiers(tries=9, retraction_depth=2)
    # Skip initial clockwork and eventually-Helly checks for an exact
    # isolation of the retraction suite.
    for callback in tests[2:]:
        callback(FakeSequence())
    assert calls[:4] == [originals] * 4
    assert calls[4:8] == [iterate1] * 4
    assert calls[8:] == ["special"]


def test_retraction_depth_bound_is_enforced():
    with pytest.raises(ValueError, match="retraction_depth"):
        small.classify_clique_behavior(nx.path_graph(3), retraction_depth=3)
