"""Compare incidence-mask clique graphs with explicit graphs and fold witnesses."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest
from pycliques import (
    ImplicitCliqueGraph,
    clique_graph,
    completely_pared_graph,
    verify_implicit_paring,
)


def test_implicit_matches_explicit_on_small_graphs() -> None:
    for graph in nx.graph_atlas_g():
        if len(graph) > 5:
            break
        if not graph:
            continue
        implicit = ImplicitCliqueGraph.from_graph(graph)
        assert implicit is not None
        explicit = clique_graph(graph)
        assert explicit is not None
        materialized = implicit.to_networkx()
        assert materialized is not None
        labels = {v: implicit.clique(v) for v in implicit}
        assert nx.utils.graphs_equal(
            nx.relabel_nodes(materialized, labels), explicit
        )
        assert implicit.number_of_edges() == explicit.number_of_edges()
        for v in implicit:
            expected = set(materialized[v]) | {v}
            assert implicit.closed_neighborhood_mask(v) == sum(
                1 << w for w in expected
            )
            assert implicit.dominated_vertices_mask(v) == sum(
                1 << w
                for w in materialized
                if w != v and set(materialized[w]) | {w} <= expected
            )
        for u in implicit:
            for v in implicit:
                assert implicit.has_edge(u, v) == materialized.has_edge(u, v)
        pared = implicit.pare()
        assert verify_implicit_paring(implicit, pared)
        core = pared.core.to_networkx()
        assert core is not None
        assert nx.is_isomorphic(core, completely_pared_graph(explicit))
        assert len(implicit) - len(pared.core) == len(pared.moves)
        assert not pared.core.pare().moves


def test_implicit_bounds_and_labels() -> None:
    graph = nx.cycle_graph(["a", "b", "c", "d"])
    assert ImplicitCliqueGraph.from_graph(graph, max_cliques=3) is None
    implicit = ImplicitCliqueGraph.from_graph(graph, max_cliques=4)
    assert implicit is not None
    assert len(implicit) == 4
    assert implicit.to_networkx(max_edges=3) is None
    assert implicit.to_networkx(max_edges=4) is not None
    assert all(isinstance(label, str) for v in implicit for label in implicit.clique(v))
    for invalid in (-1, True):
        with pytest.raises(ValueError):
            ImplicitCliqueGraph.from_graph(graph, max_cliques=invalid)
        with pytest.raises(ValueError):
            implicit.to_networkx(max_edges=invalid)
    with pytest.raises(KeyError):
        implicit.has_edge(-1, 0)
    with pytest.raises(KeyError):
        implicit.clique(4)
    with pytest.raises(KeyError):
        implicit.closed_neighborhood_mask(True)


def test_implicit_empty_graph() -> None:
    implicit = ImplicitCliqueGraph.from_graph(nx.Graph(), max_cliques=0)
    assert implicit is not None
    assert len(implicit) == 0
    assert implicit.number_of_edges() == 0
    assert verify_implicit_paring(implicit, implicit.pare())
    explicit = implicit.to_networkx(max_edges=0)
    assert explicit is not None and not explicit


def test_invalid_paring_certificates() -> None:
    graph = nx.path_graph(4)
    implicit = ImplicitCliqueGraph.from_graph(graph)
    assert implicit is not None
    result = implicit.pare()
    assert result.moves
    p, q = result.moves[0]
    for moves in (
        ((p, p),),
        ((-1, q),),
        ((p, 100),),
        ((p, True),),
        ((p, q), (p, q)),
        (),
    ):
        assert not verify_implicit_paring(implicit, replace(result, moves=moves))
    assert not verify_implicit_paring(implicit, replace(result, core=implicit))
    cycle = ImplicitCliqueGraph.from_graph(nx.cycle_graph(4))
    assert cycle is not None
    fake = replace(cycle.pare(), moves=((0, 1),))
    assert not verify_implicit_paring(cycle, fake)


@pytest.mark.slow
def test_cubic_77_large_implicit_iterate() -> None:
    graph = completely_pared_graph(
        nx.complement(nx.from_graph6_bytes(b"KsX@?oS@_D?J"))
    )
    for order in (19, 64):
        nxt = clique_graph(graph)
        assert nxt is not None
        graph = completely_pared_graph(nxt)
        assert len(graph) == order
    graph = nx.convert_node_labels_to_integers(graph)
    implicit = ImplicitCliqueGraph.from_graph(graph, max_cliques=81_000)
    assert implicit is not None
    assert len(implicit) == 80_909
    pared = implicit.pare()
    assert len(pared.core) == 76_787
    assert len(pared.moves) == 4_122
    assert pared.core.number_of_edges() == 2_896_524_461
    assert verify_implicit_paring(implicit, pared)
    assert pared.core.to_networkx(max_edges=1_000_000) is None
