"""Small exact tests and the snub disphenoid divergence certificate."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import networkx as nx
import pycliques
import pytest

DATA = Path(__file__).parent / "data" / "snub_o24_retraction.json.b85"


def _certificate() -> dict:
    """Read the zlib/base85-encoded JSON without third-party dependencies."""
    data = json.loads(zlib.decompress(base64.b85decode(DATA.read_text().strip())))
    if not isinstance(data, dict):
        raise ValueError("snub retraction certificate must be a JSON object")
    return data


def _small_fixture() -> tuple[nx.Graph, list[tuple[int, int]]]:
    """O3 with an extra vertex folded onto vertex zero."""
    host = pycliques.octahedron(3)
    host.add_node(6)
    host.add_edges_from((6, v) for v in [0, 2, 3])
    return host, [(0, 1), (2, 3), (4, 5)]


def test_induced_target_search_and_validation() -> None:
    host, pairs = _small_fixture()
    result = pycliques.find_octahedral_retraction(host, pairs, max_states=500)
    assert result.status == pycliques.RetractionSearchStatus.FOUND
    assert result.retraction is not None
    assert result.inclusion is not None
    assert pycliques.verify_retraction_witness(
        host,
        host.subgraph([v for pair in pairs for v in pair]),
        (result.retraction, result.inclusion),
    )


def test_exhausted_and_limited_are_distinct() -> None:
    host, pairs = _small_fixture()
    # A universal extra vertex cannot map into O3 when the target is fixed.
    host.add_edges_from((6, v) for v in [1, 4, 5])
    result = pycliques.find_octahedral_retraction(host, pairs)
    assert result.status == pycliques.RetractionSearchStatus.EXHAUSTED
    assert result.retraction is None

    host, pairs = _small_fixture()
    limited = pycliques.find_octahedral_retraction(host, pairs, max_states=1)
    assert limited.status == pycliques.RetractionSearchStatus.LIMIT_REACHED
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
        pycliques.find_octahedral_retraction(host, pairs)


def _canonical_clique_graph(graph: nx.Graph) -> nx.Graph:
    """Certificate labels maximal cliques by increasing vertex bitmask."""
    result = pycliques.clique_graph(graph, bound=2_000)
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
        (i, 4 + j) for i in range(4) for j in range(4) if i in (0, 3) or j in (0, 3)
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
    assert nx.is_isomorphic(target, pycliques.octahedron(24))
    projection = dict(enumerate(data["images"]))
    assert pycliques.verify_retraction_witness(
        snub_fourth_iterate, target, (projection, {v: v for v in vertices})
    )


@pytest.mark.slow
def test_snub_o24_retraction_is_rediscovered(snub_fourth_iterate: nx.Graph) -> None:
    """Optional research test: rediscover a map from the same O24 embedding."""
    pairs = _certificate()["target_pairs"]
    result = pycliques.find_octahedral_retraction(
        snub_fourth_iterate, pairs, max_states=5_000
    )
    assert result.status == pycliques.RetractionSearchStatus.FOUND
    assert result.retraction is not None
    assert result.inclusion is not None
    assert pycliques.verify_retraction_witness(
        snub_fourth_iterate,
        snub_fourth_iterate.subgraph([v for pair in pairs for v in pair]),
        (result.retraction, result.inclusion),
    )


@pytest.mark.parametrize(
    "options",
    [
        {"max_states": 0},
        {"max_states": True},
        {"max_states": 1.5},
        {"seed": 2.5},
    ],
)
def test_search_rejects_invalid_limits_and_seed(options: dict) -> None:
    with pytest.raises(ValueError):
        pycliques.find_octahedral_retraction(
            pycliques.octahedron(3), [(0, 1), (2, 3), (4, 5)], **options
        )


def test_search_validates_missing_vertices_and_cross_edges() -> None:
    pairs = [(0, 1), (2, 3), (4, 5)]
    host = pycliques.octahedron(3)
    with pytest.raises(ValueError, match="belong to the host"):
        pycliques.find_octahedral_retraction(host, [(0, 1), (2, 3), (4, 8)])
    host.remove_edge(0, 2)
    with pytest.raises(ValueError, match="completely joined"):
        pycliques.find_octahedral_retraction(host, pairs)


@pytest.mark.parametrize("graph_class", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_rejects_directed_and_multigraph_hosts(
    graph_class: type[nx.Graph],
) -> None:
    # Even reciprocal directed edges do not make a valid simple graph input.
    host = graph_class(pycliques.octahedron(3))
    with pytest.raises(ValueError, match="undirected simple loopless graph"):
        pycliques.find_octahedral_retraction(host, [(0, 1), (2, 3), (4, 5)])


def test_rejects_looped_host() -> None:
    host = pycliques.octahedron(3)
    host.add_edge(0, 0)
    with pytest.raises(ValueError, match="undirected simple loopless graph"):
        pycliques.find_octahedral_retraction(host, [(0, 1), (2, 3), (4, 5)])


def test_target_itself_needs_no_search_branches() -> None:
    host = pycliques.octahedron(3)
    result = pycliques.find_octahedral_retraction(
        host, [(0, 1), (2, 3), (4, 5)], max_states=1
    )
    assert result.status is pycliques.RetractionSearchStatus.FOUND
    assert result.states_explored == 1
    assert result.assignments_tried == 0
    assert result.retraction == {v: v for v in host}


def test_initial_singleton_propagation_rejects_opposite_images() -> None:
    host = pycliques.octahedron(3)
    host.add_nodes_from(("left", "right"))
    # The two added vertices are initially forced to antipodal images
    # 0 and 1, but they are adjacent in the host.
    host.add_edges_from(("left", v) for v in (0, 2, 3, 4, 5))
    host.add_edges_from(("right", v) for v in (1, 2, 3, 4, 5))
    host.add_edge("left", "right")
    result = pycliques.find_octahedral_retraction(host, [(0, 1), (2, 3), (4, 5)])
    assert result.status is pycliques.RetractionSearchStatus.EXHAUSTED
    assert result.states_explored == 1
    assert result.assignments_tried == 0


@pytest.mark.parametrize(
    "budget,expected_status,attempted",
    [
        (1, pycliques.RetractionSearchStatus.LIMIT_REACHED, 0),
        (2, pycliques.RetractionSearchStatus.LIMIT_REACHED, 1),
        (3, pycliques.RetractionSearchStatus.LIMIT_REACHED, 2),
        (4, pycliques.RetractionSearchStatus.FOUND, 3),
    ],
)
def test_rejected_propagation_counts_toward_budget(
    budget: int, expected_status: pycliques.RetractionSearchStatus, attempted: int
) -> None:
    # This 10-vertex graph provokes an immediately rejected search branch.
    # Previously the rejection was omitted from states_explored, so the
    # reported number of states could be smaller than attempted + 1.
    host = nx.from_graph6_bytes(b"I]~qs|njw")
    result = pycliques.find_octahedral_retraction(
        host, [(0, 1), (2, 3), (4, 5)], max_states=budget
    )
    assert result.status is expected_status
    assert result.assignments_tried == attempted
    assert result.states_explored == attempted + 1
    assert result.states_explored <= budget
    if result.status is pycliques.RetractionSearchStatus.FOUND:
        assert result.retraction is not None
        assert result.inclusion is not None
        assert pycliques.verify_retraction_witness(
            host,
            host.subgraph(range(6)),
            (result.retraction, result.inclusion),
        )


def test_solver_rejects_its_own_invalid_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pycliques.octahedral_retractions.verify_retraction_witness",
        lambda *_: False,
    )
    with pytest.raises(AssertionError, match="invalid witness"):
        pycliques.find_octahedral_retraction(
            pycliques.octahedron(3), [(0, 1), (2, 3), (4, 5)]
        )
