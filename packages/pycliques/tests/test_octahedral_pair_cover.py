"""Exact covering bounds must never discard an octahedral completion."""

from __future__ import annotations

from itertools import combinations
import random

import networkx as nx

from pycliques.octahedral_blind import (
    OctahedralBlindStatus,
    _OctahedralMasks,
    complete_dominating_octahedron,
)
from pycliques.octahedral_pair_cover import pair_cover_obstructs


def _vertices(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask -= bit


def _independent_completion_oracle(nonedge, common: int, missing: int) -> bool:
    """Exhaustively inspect all compatible missing pairs on tiny graphs."""
    if missing == 0:
        return common == 0
    for a in _vertices(common):
        for b in _vertices(nonedge[a] & common):
            if b > a:
                next_common = common & ~(nonedge[a] | nonedge[b])
                if _independent_completion_oracle(
                    nonedge, next_common, missing - 1
                ):
                    return True
    return False


def test_unweighted_cover_excludes_when_basic_checks_do_not():
    graph = nx.from_graph6_bytes(b"On^K~I^~o|{~^~ervK~r~")
    masks = _OctahedralMasks(graph)
    common = masks.common(((5, 13),))
    assert common.bit_count() == 7
    assert all(masks.nonedge[v] & common for v in _vertices(common))
    assert pair_cover_obstructs(masks.nonedge, common, 1)
    assert not _independent_completion_oracle(masks.nonedge, common, 1)
    assert (
        complete_dominating_octahedron(
            graph, [(5, 13)], target_dimension=2, max_states=1000
        ).status
        is OctahedralBlindStatus.EXHAUSTED
    )


def test_integer_weights_can_exclude_when_uniform_weights_do_not():
    graph = nx.from_graph6_bytes(
        rb"Y~r{u|Vr~~zvn~~nl~v|^yvfq}|jlt~t~|V^Vw^vJlf~z~v\}n~~V~v_"
    )
    masks = _OctahedralMasks(graph)
    common = masks.common(((0, 8),))
    eligible = [
        (a, b)
        for a in _vertices(common)
        for b in _vertices(masks.nonedge[a] & common)
        if a < b
    ]
    assert common.bit_count() == 15 and len(eligible) == 24
    gains = sorted(
        (
            ((masks.nonedge[a] | masks.nonedge[b]) & common).bit_count()
            for a, b in eligible
        ),
        reverse=True,
    )
    assert sum(gains[:2]) >= common.bit_count()
    assert pair_cover_obstructs(masks.nonedge, common, 2)
    assert not _independent_completion_oracle(masks.nonedge, common, 2)


def test_positive_oct4_prefix_survives():
    graph = nx.complete_multipartite_graph(2, 2, 2, 2)
    masks = _OctahedralMasks(graph)
    for count in range(1, 4):
        common = masks.common(tuple((2 * i, 2 * i + 1) for i in range(count)))
        assert not pair_cover_obstructs(masks.nonedge, common, 4 - count)


def test_random_small_cover_rejections_against_exhaustive_oracle():
    rng = random.Random(1598)
    rejected = 0
    checked = 0
    for _ in range(75):
        n = rng.randrange(6, 10)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for u, v in combinations(range(n), 2):
            if rng.random() < 0.5:
                graph.add_edge(u, v)
        masks = _OctahedralMasks(graph)
        for u, v in combinations(range(n), 2):
            if graph.has_edge(u, v):
                continue
            common = masks.common(((u, v),))
            for missing in (1, 2):
                if pair_cover_obstructs(masks.nonedge, common, missing):
                    rejected += 1
                    assert not _independent_completion_oracle(
                        masks.nonedge, common, missing
                    )
                checked += 1
    assert checked > 1000
    assert rejected > 0
