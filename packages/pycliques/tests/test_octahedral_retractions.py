"""Small exact tests and the snub disphenoid divergence certificate."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import networkx as nx
import pytest

from pycliques import (
    RetractionSearchStatus,
    clique_graph,
    find_octahedral_retraction,
    octahedron,
    verify_retraction_witness,
)


DATA = Path(__file__).parent / "data" / "snub_o24_retraction.json.b85"


def _certificate() -> dict:
    """Read the zlib/base85-encoded JSON without third-party dependencies."""
    return json.loads(zlib.decompress(base64.b85decode(DATA.read_text().strip())))


def _small_fixture() -> tuple[nx.Graph, list[tuple[int, int]]]:
    """O3 with an extra vertex folded onto vertex zero."""
    host = octahedron(3)
    host.add_node(6)
    host.add_edges_from((6, v) for v in [0, 2, 3])
    return host, [(0, 1), (2, 3), (4, 5)]


def test_induced_target_search_and_validation() -> None:
    host, pairs = _small_fixture()
    result = find_octahedral_retraction(host, pairs, max_states=500)
    assert result.status == RetractionSearchStatus.FOUND
    assert result.retraction is not None
    assert result.inclusion is not None
    assert verify_retraction_witness(
        host,
        host.subgraph([v for pair in pairs for v in pair]),
        (result.retraction, result.inclusion),
    )


def test_exhausted_and_limited_are_distinct() -> None:
    host, pairs = _small_fixture()
    # A universal extra vertex cannot map into O3 when the target is fixed.
    host.add_edges_from((6, v) for v in [1, 4, 5])
    result = find_octahedral_retraction(host, pairs)
    assert result.status == RetractionSearchStatus.EXHAUSTED
    assert result.retraction is None

    host, pairs = _small_fixture()
    limited = find_octahedral_retraction(host, pairs, max_states=1)
    assert limited.status == RetractionSearchStatus.LIMIT_REACHED
    assert limited.retraction is None


@pytest.mark.parametrize(
    "pairs",
    [
        [(0, 1), (2, 3)],
        [(0, 1), (2, 3), (4, 4)],
        [(0, 2), (1, 3), (4, 5)],
    ],
)
def test_invalid_target(pairs: list[tuple[int, int]]) -> None:
    host, _ = _small_fixture()
    with pytest.raises(ValueError):
        find_octahedral_retraction(host, pairs)


def _canonical_clique_graph(graph: nx.Graph) -> nx.Graph:
    """Certificate labels maximal cliques by increasing vertex bitmask."""
    result = clique_graph(graph, bound=2_000)
    assert result is not None
    order = sorted(result, key=lambda clique: sum(1 << int(v) for v in clique))
    return nx.relabel_nodes(result, {clique: i for i, clique in enumerate(order)})


@pytest.fixture(scope="module")
def snub_fourth_iterate() -> nx.Graph:
    # The saved certificate uses F_{4,4} labels, not graph6 labels.
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from((i, i + 1) for i in range(3))
    graph.add_edges_from((i, i + 1) for i in range(4, 7))
    graph.add_edges_from(
        (i, 4 + j) for i in range(4) for j in range(4)
        if i in (0, 3) or j in (0, 3)
    )
    assert nx.is_isomorphic(graph, nx.from_graph6_bytes(b"GQyuzw"))
    expected = [(8, 18), (12, 46), (20, 136), (56, 1196), (1076, 504360)]
    for i, (vertices, edges) in enumerate(expected):
        assert (len(graph), graph.number_of_edges()) == (vertices, edges)
        if i < 4:
            graph = _canonical_clique_graph(graph)
    return graph


def test_snub_o24_certificate_regression(snub_fourth_iterate: nx.Graph) -> None:
    """Replay the positive certificate in normal CI without any search."""
    data = _certificate()
    pairs = data["target_pairs"]
    vertices = [v for pair in pairs for v in pair]
    target = snub_fourth_iterate.subgraph(vertices)
    assert nx.is_isomorphic(target, octahedron(24))
    projection = dict(enumerate(data["images"]))
    assert verify_retraction_witness(
        snub_fourth_iterate, target, (projection, {v: v for v in vertices})
    )


@pytest.mark.slow
def test_snub_o24_retraction_is_rediscovered(snub_fourth_iterate: nx.Graph) -> None:
    """Optional research test: rediscover a map from the same O24 embedding."""
    pairs = _certificate()["target_pairs"]
    result = find_octahedral_retraction(snub_fourth_iterate, pairs, max_states=5_000)
    assert result.status == RetractionSearchStatus.FOUND
    assert result.retraction is not None
    assert result.inclusion is not None
    assert verify_retraction_witness(
        snub_fourth_iterate,
        snub_fourth_iterate.subgraph([v for pair in pairs for v in pair]),
        (result.retraction, result.inclusion),
    )
