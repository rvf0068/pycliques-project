"""Exercise certified reductions on meaningful success and rejection paths."""

from dataclasses import replace

import networkx as nx
import pytest
from pycombtop import (
    FlagMove,
    FlagReductionCertificate,
    reduce_flag_graph,
    reduced_star_wedge_certificate,
    verify_flag_reduction,
    verify_reduced_star_wedge,
    verify_vertex_star_wedge,
    vertex_star_wedge_certificate,
)
from pycombtop.flag_edge_star import (
    _cone_fold_passes,
    _contractible_edge_passes,
    _dong_wedge,
)


def _two_apices_over_path() -> nx.Graph:
    """Edge (4,5) has contractible link P4, but no cone apex."""
    graph = nx.path_graph(4)
    graph.add_edge(4, 5)
    graph.add_edges_from((u, v) for u in (4, 5) for v in range(4))
    return graph


def _two_spheres_at_vertex() -> nx.Graph:
    """A flag complex with deletion S2 and link S1 at vertex 6."""
    graph = nx.octahedral_graph()
    # 0,5 and 1,4 are the two nonedges of the induced equatorial C4.
    graph.add_edges_from((6, u) for u in (0, 5, 1, 4))
    return graph


def test_dense_passes_and_replay():
    graph = nx.complete_graph(5)
    core, moves = _cone_fold_passes(graph, passes=3, max_moves=100)
    assert moves
    assert all(m.kind in {"cone_edge", "fold_vertex"} for m in moves)
    proof = FlagReductionCertificate(tuple(moves), tuple(core), tuple(core.edges))
    assert verify_flag_reduction(graph, proof)

    # Triangle-free graphs have no cone-link edge, and no fold on a cycle.
    cycle = nx.cycle_graph(6)
    cycle_core, cycle_moves = _cone_fold_passes(
        cycle, passes=3, max_moves=10
    )
    assert not cycle_moves
    assert nx.is_isomorphic(cycle_core, cycle)


def test_contractible_link_search_with_exact_replay():
    graph = _two_apices_over_path()
    core, moves = _contractible_edge_passes(
        graph, passes=2, max_moves=100, max_link_vertices=20, max_nodes=30_000
    )
    assert moves
    assert all(move.kind == "contractible_edge" for move in moves)
    proof = FlagReductionCertificate(tuple(moves), tuple(core), tuple(core.edges))
    assert verify_flag_reduction(graph, proof)
    with pytest.raises(OverflowError, match="move limit"):
        _contractible_edge_passes(
            graph, passes=1, max_moves=0, max_link_vertices=20,
            max_nodes=30_000
        )
    # Forbid larger common links without claiming a negative topological result.
    untouched, no_moves = _contractible_edge_passes(
        nx.cycle_graph(5), passes=2, max_moves=10,
        max_link_vertices=1, max_nodes=30_000
    )
    assert not no_moves
    assert untouched.number_of_edges() == 5


def test_bad_reduction_proofs_fail_closed():
    graph = nx.complete_graph(3)
    empty = FlagReductionCertificate((), tuple(graph), tuple(graph.edges))
    assert verify_flag_reduction(graph, empty)
    assert not verify_flag_reduction(
        graph, replace(empty, remaining_vertices=(0, 1))
    )
    assert not verify_flag_reduction(
        graph, replace(empty, remaining_edges=((0, 1),))
    )
    bad = [
        FlagMove("unknown_kind", 0, 1),
        FlagMove("cone_edge", 0, 1, 99),
        FlagMove("cone_edge", 0, 99, 2),
        FlagMove("fold_vertex", 0, 99),
        FlagMove("contractible_edge", 0, 99),
        FlagMove("cone_edge", 0, 1, []),
    ]
    for move in bad:
        assert not verify_flag_reduction(graph, replace(empty, moves=(move,)))
    path = nx.path_graph(4)
    invalid_fold = FlagReductionCertificate(
        (FlagMove("fold_vertex", 1, 2),),
        tuple(path), tuple(path.edges),
    )
    assert not verify_flag_reduction(path, invalid_fold)
    invalid_link = FlagReductionCertificate(
        (FlagMove("contractible_edge", 0, 1),),
        tuple(path), tuple(path.edges),
    )
    assert not verify_flag_reduction(path, invalid_link)
    assert not verify_flag_reduction(nx.DiGraph([(0, 1)]), empty)
    assert not verify_flag_reduction(nx.MultiGraph([(0, 1)]), empty)


def test_vertex_star_exact_small_wedge_and_wrapper():
    graph = _two_spheres_at_vertex()
    result = vertex_star_wedge_certificate(graph, candidates=[6])
    assert result is not None
    assert result.deletion_spheres == {2: 1}
    assert result.link_spheres == {1: 1}
    assert result.spheres == {2: 2}
    assert verify_vertex_star_wedge(graph, result)

    # No cone-edge or fold exists in this small equatorial construction.
    whole = reduced_star_wedge_certificate(
        graph, candidates=[6], cone_passes=1, contractible_passes=0
    )
    assert whole is not None
    assert whole.spheres == {2: 2}
    assert verify_reduced_star_wedge(graph, whole)
    assert not verify_reduced_star_wedge(
        graph, replace(whole, spheres={2: 3})
    )
    wrong_reduction = replace(
        whole.reduction, remaining_vertices=(0, 1)
    )
    assert not verify_reduced_star_wedge(
        graph, replace(whole, reduction=wrong_reduction)
    )
    assert not verify_vertex_star_wedge(
        graph, replace(result, vertex=999)
    )


def test_vertex_star_inconclusive_conditions():
    assert _dong_wedge(nx.path_graph(4)) == ({}, {})
    assert _dong_wedge(nx.cycle_graph(4)) == ({1: 1}, {1: 1})
    assert _dong_wedge(nx.empty_graph(3)) is None
    assert _dong_wedge(nx.complete_graph(81)) is None

    graph = _two_spheres_at_vertex()
    assert vertex_star_wedge_certificate(graph, candidates=[999]) is None
    assert vertex_star_wedge_certificate(graph, max_candidates=0) is None
    assert vertex_star_wedge_certificate(nx.empty_graph(81)) is None
    assert vertex_star_wedge_certificate(nx.path_graph(3), candidates=[1]) is None

    # Here the deletion is S1, not simply connected: the gap test must refuse.
    cone_cycle = nx.cycle_graph(4)
    cone_cycle.add_edges_from((4, i) for i in range(4))
    assert vertex_star_wedge_certificate(cone_cycle, candidates=[4]) is None
    assert not verify_vertex_star_wedge(
        cone_cycle,
        replace(
            vertex_star_wedge_certificate(graph, candidates=[6]),
            vertex=4,
        ),
    )
    # A disconnected two-vertex link is outside the connected-link proof.
    disconnected_link = nx.octahedral_graph()
    disconnected_link.add_edges_from(((6, 0), (6, 5)))
    assert vertex_star_wedge_certificate(
        disconnected_link, candidates=[6]
    ) is None


def test_opt_in_resource_rejections_and_empty_search():
    assert reduce_flag_graph(nx.empty_graph(1001)) is None
    assert reduced_star_wedge_certificate(nx.empty_graph(1001)) is None
    assert reduced_star_wedge_certificate(nx.Graph()) is None
    graph = nx.cycle_graph(5)
    assert reduce_flag_graph(graph, cone_passes=0) is None
    assert reduce_flag_graph(graph, contractible_passes=-1) is None
    assert reduce_flag_graph(graph, max_contractible_vertices=-1) is None
    assert reduce_flag_graph(graph, max_link_vertices=0) is None
    assert reduce_flag_graph(graph, max_nodes=0) is None
    assert reduce_flag_graph(graph, max_moves=0) is None
    assert reduce_flag_graph(graph, max_vertices=4) is None
    core, proof = reduce_flag_graph(graph, cone_passes=2)
    assert len(core) == 5
    assert verify_flag_reduction(graph, proof)
