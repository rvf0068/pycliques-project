import networkx as nx
from pycliques.cutpoints import (
    cutpoint_edge_contractions,
    cutpoint_edge_removals,
    cutpoint_reductions,
    has_local_cutpoints,
    local_cutpoints,
    neighborhood_components,
    reduction_retracts_to,
)


def test_local_cutpoints_path():
    """Interior vertices of a path are local cutpoints."""
    assert sorted(local_cutpoints(nx.path_graph(5))) == [1, 2, 3]


def test_local_cutpoints_complete_graph():
    """Complete graphs have no local cutpoints."""
    assert list(local_cutpoints(nx.complete_graph(5))) == []


def test_local_cutpoints_cycle():
    """Triangles (K_3) have no local cutpoints, but longer cycles do."""
    assert list(local_cutpoints(nx.cycle_graph(3))) == []
    assert len(list(local_cutpoints(nx.cycle_graph(6)))) == 6


def test_local_cutpoints_single_vertex():
    """A single vertex has no local cutpoints."""
    assert list(local_cutpoints(nx.trivial_graph())) == []


def test_has_local_cutpoints_true():
    """Path graphs have local cutpoints."""
    assert has_local_cutpoints(nx.path_graph(4)) is True


def test_has_local_cutpoints_false():
    """Complete graphs have no local cutpoints."""
    assert has_local_cutpoints(nx.complete_graph(4)) is False


def test_neighborhood_components_path_interior():
    """Interior vertex of a path has two singleton components."""
    comps = neighborhood_components(nx.path_graph(5), 2)
    assert len(comps) == 2
    assert {frozenset(c) for c in comps} == {frozenset({1}), frozenset({3})}


def test_neighborhood_components_complete_graph():
    """In a complete graph, N(v) is connected, so one component."""
    comps = neighborhood_components(nx.complete_graph(5), 0)
    assert len(comps) == 1


def _bridge_of_two_c5() -> nx.Graph:
    """Two C_5 copies connected by a single bridge edge."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    g.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9), (9, 5)])
    g.add_edge(0, 5)
    return g


def test_cutpoint_edge_removals_path():
    """Edge removal at cutpoints of a bridge graph yields C_5 pieces."""
    results = list(cutpoint_edge_removals(_bridge_of_two_c5()))
    assert len(results) > 0
    for g in results:
        assert g.order() >= 2


def test_cutpoint_edge_removals_complete_graph():
    """Complete graph has no cutpoints, so no reductions."""
    assert list(cutpoint_edge_removals(nx.complete_graph(4))) == []


def test_cutpoint_edge_contractions_path():
    """Edge contraction at cutpoints of a bridge graph yields subgraphs."""
    results = list(cutpoint_edge_contractions(_bridge_of_two_c5()))
    assert len(results) > 0


def test_cutpoint_edge_contractions_complete_graph():
    """Complete graph has no cutpoints, so no contractions."""
    assert list(cutpoint_edge_contractions(nx.complete_graph(4))) == []


def test_cutpoint_reductions_complete_graph_empty():
    """No reductions for a complete graph."""
    assert list(cutpoint_reductions(nx.complete_graph(4))) == []


def test_cutpoint_reductions_yields_from_both():
    """cutpoint_reductions yields results from both removal and contraction."""
    g = _bridge_of_two_c5()
    removals = list(cutpoint_edge_removals(g))
    contractions = list(cutpoint_edge_contractions(g))
    all_reductions = list(cutpoint_reductions(g))
    assert len(all_reductions) == len(removals) + len(contractions)


def test_reduction_retracts_to_path():
    """A bridge-of-two-C5s reduction yields C_5, which retracts to C_5."""
    assert reduction_retracts_to(_bridge_of_two_c5(), nx.cycle_graph(5)) is True


def test_reduction_retracts_to_false():
    """Complete graph has no cutpoints, nothing retracts."""
    assert reduction_retracts_to(nx.complete_graph(4), nx.cycle_graph(4)) is False


def test_local_cutpoints_wheel():
    """Hub of a wheel graph is not a local cutpoint (neighborhood is a cycle)."""
    w = nx.wheel_graph(5)  # hub is vertex 0
    cutpts = list(local_cutpoints(w))
    assert 0 not in cutpts
