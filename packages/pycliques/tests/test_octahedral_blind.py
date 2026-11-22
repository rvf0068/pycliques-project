"""Exact screening, completion, hitting-set, arbitrary labels, and blind search."""

import random

import networkx as nx
import pytest

from pycliques.octahedral_blind import (
    OctahedralBlindStatus,
    complete_dominating_octahedron,
    minimum_octahedral_repair,
    minimize_octahedral_conflict,
    screen_partial_octahedral_retraction,
)
from pycliques.octahedral_blind_search import find_blind_octahedral_retraction
from pycliques.octahedral_joint_fibers import discover_partial_octahedral_fibers
from pycliques.octahedral_nonlocal_search import verify_induced_octahedron


def octahedron(d):
    return nx.complete_multipartite_graph(*([2]*d))


def test_small_blind_retraction():
    graph = octahedron(3)
    result = find_blind_octahedral_retraction(
        graph, target_dimension=3, seed=11, delete_count=0,
        core_rounds=4, completion_states=200, retraction_states=500,
    )
    assert result.status is OctahedralBlindStatus.FOUND
    assert set(result.projection) == set(graph)


def test_arbitrary_vertex_labels_and_completion():
    graph = nx.relabel_nodes(
        octahedron(4), {x:("a",x) for x in range(8)}
    )
    pairs = [("a",0),("a",1)], [("a",2),("a",3)]
    completion = complete_dominating_octahedron(
        graph, pairs, target_dimension=4, max_states=10000
    )
    assert completion.status is OctahedralBlindStatus.FOUND
    assert verify_induced_octahedron(
        graph, completion.antipodal_pairs, require_dominating=True
    )
    assert screen_partial_octahedral_retraction(graph, pairs).possible


def test_completion_reports_exhausted_and_limit():
    graph = octahedron(3)
    pairs = [(0,1),(2,3)]
    assert complete_dominating_octahedron(
        graph,pairs,target_dimension=4,max_states=10000
    ).status is OctahedralBlindStatus.EXHAUSTED
    assert complete_dominating_octahedron(
        graph,pairs,target_dimension=3,max_states=1
    ).status is OctahedralBlindStatus.LIMIT_REACHED


def test_partial_screen_forced_edge_and_conflict():
    graph = octahedron(3)
    graph.add_nodes_from(("left", "right"))
    graph.add_edges_from(("left",x) for x in (0,2,3,4,5))
    graph.add_edges_from(("right",x) for x in (1,2,3,4,5))
    graph.add_edge("left","right")
    pairs = [(0,1),(2,3),(4,5)]
    assert screen_partial_octahedral_retraction(graph,pairs).reason == "FORCED_EDGE"
    conflict = minimize_octahedral_conflict(graph, pairs)
    assert conflict is not None
    assert conflict.deletion_minimal
    assert not screen_partial_octahedral_retraction(
        graph, conflict.antipodal_pairs
    ).possible
    for i in range(len(conflict.antipodal_pairs)):
        small = conflict.antipodal_pairs[:i] + conflict.antipodal_pairs[i+1:]
        assert screen_partial_octahedral_retraction(graph, small).possible


def test_weighted_hitting_set_against_exhaustive():
    rng=random.Random(501)
    for k in range(1,7):
        for _ in range(24):
            cores=[rng.randint(1,(1<<k)-1) for _ in range(9)]
            weights=[rng.randint(1,5) for _ in range(k)]
            result=minimum_octahedral_repair(cores,k,weights=weights)
            best=min(sum(weights[i] for i in range(k) if mask&(1<<i))
                     for mask in range(1<<k)
                     if all(mask & c for c in cores))
            assert result.cost == best
            chosen=sum(1<<i for i in result.deleted_indices)
            assert all(chosen & core for core in cores)


def test_positive_random_retract_prefixes():
    rng=random.Random(159)
    for _ in range(70):
        d=rng.choice((3,4,5))
        n=2*d+rng.randint(1,8)
        graph=octahedron(d)
        graph.add_nodes_from(range(2*d,n))
        pairs=[(2*i,2*i+1) for i in range(d)]
        images=list(range(2*d))+[rng.randrange(2*d) for _ in range(n-2*d)]
        for u in range(n):
            for v in range(u+1,n):
                if images[u] ^ 1 == images[v]:
                    continue
                if u < 2*d and v<2*d or rng.random()<.72:
                    graph.add_edge(u,v)
        for k in range(2,d+1):
            assert screen_partial_octahedral_retraction(graph,pairs[:k]).possible


def test_partial_fiber_search_returns_verifiable_map():
    graph=octahedron(4)
    result=discover_partial_octahedral_fibers(
        graph,4,seed=71,beam=8,samples=12
    )
    assert 3<=len(result.antipodal_pairs)<=4
    assert result.covered_vertices==len(result.projection)
    anchors={v for p in result.antipodal_pairs for v in p}
    assert all(result.projection[x]==x for x in anchors)
    assert all(result.projection[u]==result.projection[v] or
               graph.has_edge(result.projection[u],result.projection[v])
        for u, v in graph.edges
        if u in result.projection and v in result.projection
    )


@pytest.mark.parametrize("bad",[0,True,-1])
def test_invalid_limit(bad):
    with pytest.raises(ValueError):
        complete_dominating_octahedron(octahedron(3), [], target_dimension=3,
                                       max_states=bad)
