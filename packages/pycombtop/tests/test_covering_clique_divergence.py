"""Independent integration certificates for a triangular double cover."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from itertools import combinations

import networkx as nx
import pytest
from pycliques.cliques import Clique, clique_graph
from pycliques.clockwork_pairs import clockwork_coaffine_pair
from pycliques.clockwork_search import CertificateVerificationStatus
from pycliques.small import Certificate
from pycombtop.covering_divergence import (
    TriangularCoverDivergenceCertificate,
    verify_triangular_cover_divergence_certificate,
)
from pycombtop.fundamental_group import covering_graph, fundamental_group

VOLTAGE_ONE = frozenset(
    frozenset(edge)
    for edge in (
        (1, 5),
        (2, 5),
        (2, 6),
        (2, 8),
        (4, 8),
        (5, 9),
    )
)
F_INDEX = {
    0: (1, 0),
    1: (5, 0),
    2: (3, 0),
    3: (11, 1),
    4: (6, 1),
    5: (7, 1),
    6: (5, 1),
    7: (1, 1),
    8: (11, 0),
    9: (3, 1),
    10: (7, 0),
    11: (6, 0),
    12: (0, 0),
    13: (9, 1),
    14: (4, 1),
    15: (0, 1),
    16: (9, 0),
    17: (4, 0),
}


def voltage(edge: tuple[int, int], cocycle: frozenset[frozenset[int]]) -> int:
    return int(frozenset(edge) in cocycle)


def explicit_double_cover(
    graph: nx.Graph, cocycle: frozenset[frozenset[int]]
) -> nx.Graph:
    cover = nx.Graph()
    cover.add_nodes_from((vertex, sheet) for vertex in graph for sheet in (0, 1))
    for first, second in graph.edges():
        change = voltage((first, second), cocycle)
        for sheet in (0, 1):
            cover.add_edge((first, sheet), (second, sheet ^ change))
    return cover


def triangles(graph: nx.Graph) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        triangle
        for triangle in combinations(graph.nodes(), 3)
        if all(graph.has_edge(*edge) for edge in combinations(triangle, 2))
    )


def is_flat(graph: nx.Graph, cocycle: frozenset[frozenset[int]]) -> bool:
    return all(
        voltage((first, second), cocycle)
        ^ voltage((second, third), cocycle)
        ^ voltage((first, third), cocycle)
        == 0
        for first, second, third in triangles(graph)
    )


def assert_triangular_cover(
    cover: nx.Graph,
    base: nx.Graph,
    projection: dict[tuple[int, int], int],
) -> None:
    assert set(cover) == set(projection)
    assert set(projection.values()) == set(base)
    assert all(
        sum(image == vertex for image in projection.values()) == 2 for vertex in base
    )
    for vertex in cover:
        image = projection[vertex]
        closed_cover = {vertex, *cover.neighbors(vertex)}
        closed_base = {image, *base.neighbors(image)}
        projected = [projection[neighbor] for neighbor in closed_cover]
        assert len(set(projected)) == len(closed_cover)
        assert set(projected) == closed_base
        for first, second in combinations(closed_cover, 2):
            assert cover.has_edge(first, second) == base.has_edge(
                projection[first], projection[second]
            )


def sheet_gauge(base: nx.Graph, cover: nx.Graph) -> dict[int, int]:
    """Solve the projection-preserving bit gauge from the reference cover."""
    assert set(cover) == {(vertex, sheet) for vertex in base for sheet in (0, 1)}
    root = min(base)
    gauge = {root: 0}
    todo = deque((root,))
    while todo:
        first = todo.popleft()
        for second in base.neighbors(first):
            lifted = [
                sheet for sheet in (0, 1) if cover.has_edge((first, 0), (second, sheet))
            ]
            assert len(lifted) == 1
            difference = lifted[0] ^ voltage((first, second), VOLTAGE_ONE)
            expected = gauge[first] ^ difference
            if second in gauge:
                assert gauge[second] == expected
            else:
                gauge[second] = expected
                todo.append(second)
    assert set(gauge) == set(base)
    for first, second in base.edges():
        for sheet in (0, 1):
            assert cover.has_edge(
                (first, sheet ^ gauge[first]),
                (
                    second,
                    sheet ^ voltage((first, second), VOLTAGE_ONE) ^ gauge[second],
                ),
            )
    return gauge


@pytest.fixture(scope="module")
def divergence_fixture():
    graph = nx.from_graph6_bytes(b"IEjdnpy^O")
    rec = fundamental_group(graph)
    cover = covering_graph(graph, [[1, 1]], rec=rec, max_cosets=10)
    gauge = sheet_gauge(graph, cover)
    projection = {vertex: vertex[0] for vertex in cover}
    deck = {vertex: (vertex[0], 1 - vertex[1]) for vertex in cover}
    base_cliques = sorted(
        (frozenset(clique) for clique in nx.find_cliques(graph)),
        key=lambda clique: tuple(sorted(clique)),
    )

    def lifted_clique(index: int, sheet: int) -> Clique:
        base_clique = base_cliques[index]
        anchor = min(base_clique)
        voltage_lift = {
            (vertex, sheet ^ voltage((anchor, vertex), VOLTAGE_ONE))
            for vertex in base_clique
        }
        return Clique(
            (vertex, cover_sheet ^ gauge[vertex])
            for vertex, cover_sheet in voltage_lift
        )

    clique_cover = clique_graph(cover, bound=30)
    assert clique_cover is not None
    source_pair = clockwork_coaffine_pair(3, 0)
    mapping = {
        vertex: lifted_clique(index, sheet)
        for vertex, (index, sheet) in F_INDEX.items()
    }
    target_deck = {
        clique: Clique(deck[vertex] for vertex in clique) for clique in clique_cover
    }
    theorem_certificate = Certificate(
        rule="theorem_2_6_clockwork_pair_map",
        target=clique_cover,
        target_status="proven_divergent",
        map=mapping,
        m=3,
        n=0,
        radius=4,
        source=source_pair.graph,
        source_coaffination=source_pair.coaffination,
        target_coaffination=target_deck,
        theorem_base_graph=clique_cover,
    )
    composite = TriangularCoverDivergenceCertificate(
        base_graph=graph,
        cover_graph=cover,
        projection=projection,
        deck_transformation=deck,
        theorem_certificate=theorem_certificate,
        subgroup_words=((1, 1),),
        clique_bound=30,
        clique_max_edges=2000,
        deck_radius=4,
    )
    return graph, rec, cover, projection, deck, clique_cover, composite


def test_subgroup_cover_is_the_expected_triangular_voltage_cover(
    divergence_fixture,
) -> None:
    graph, rec, cover, projection, deck, clique_cover, _ = divergence_fixture
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (10, 27)
    assert len(triangles(graph)) == 19
    assert is_flat(graph, VOLTAGE_ONE)
    assert rec.rank == 1
    assert len(rec.generators) == 1
    assert not rec.relators
    assert (cover.number_of_nodes(), cover.number_of_edges()) == (20, 54)
    assert nx.is_connected(cover)

    reference = explicit_double_cover(graph, VOLTAGE_ONE)
    assert_triangular_cover(
        reference, graph, {vertex: vertex[0] for vertex in reference}
    )
    assert_triangular_cover(cover, graph, projection)
    gauge = sheet_gauge(graph, cover)
    gauge_isomorphism = {
        (vertex, sheet): (vertex, sheet ^ gauge[vertex]) for vertex, sheet in reference
    }
    assert set(gauge_isomorphism.values()) == set(cover)
    assert all(
        cover.has_edge(gauge_isomorphism[first], gauge_isomorphism[second])
        for first, second in reference.edges()
    )
    assert nx.is_isomorphic(reference, cover)
    assert nx.is_isomorphic(cover, reference)
    assert all(
        nx.shortest_path_length(cover, (vertex, 0), (vertex, 1)) == 4
        for vertex in graph
    )
    assert all(deck[deck[vertex]] == vertex for vertex in cover)
    assert (clique_cover.number_of_nodes(), clique_cover.number_of_edges()) == (
        26,
        96,
    )


def test_clockwork_pair_map_proves_divergence_through_the_verified_cover(
    divergence_fixture,
) -> None:
    *_, composite = divergence_fixture
    result = verify_triangular_cover_divergence_certificate(composite)

    assert result.status is CertificateVerificationStatus.VERIFIED, result.reason


@pytest.mark.parametrize(
    "damage",
    [
        "projection",
        "subgroup",
        "map",
        "equivariance",
        "target-missing",
        "target-extra",
    ],
)
def test_composite_verifier_rejects_tampered_provenance(
    divergence_fixture, damage: str
) -> None:
    *_, composite = divergence_fixture
    if damage == "projection":
        projection = dict(composite.projection)
        vertex = next(iter(projection))
        projection[vertex] = (
            projection[vertex] + 1
        ) % composite.base_graph.number_of_nodes()
        changed = replace(composite, projection=projection)
    elif damage == "subgroup":
        changed = replace(composite, subgroup_words=((1,),))
    elif damage in {"map", "equivariance", "target-missing", "target-extra"}:
        inner = composite.theorem_certificate
        if damage in {"map", "equivariance"}:
            mapping = dict(inner.map)
            if damage == "map":
                mapping[0] = next(
                    clique for clique in inner.target if clique != mapping[0]
                )
            else:
                mapping[1] = mapping[0]
            changed = replace(
                composite,
                theorem_certificate=replace(inner, map=mapping),
            )
        else:
            target = inner.target.copy()
            if damage == "target-missing":
                first, second = next(iter(target.edges()))
                target.remove_edge(first, second)
            else:
                nodes = tuple(target)
                missing = next(
                    (first, second)
                    for first, second in combinations(nodes, 2)
                    if not target.has_edge(first, second)
                )
                target.add_edge(*missing)
            changed = replace(
                composite,
                theorem_certificate=replace(inner, target=target),
            )
    else:
        raise AssertionError("unhandled test case")

    result = verify_triangular_cover_divergence_certificate(changed)
    assert result.status is CertificateVerificationStatus.INVALID


def test_nonflat_voltage_is_rejected() -> None:
    graph = nx.from_graph6_bytes(b"IEjdnpy^O")
    damaged = set(VOLTAGE_ONE)
    edge = next(iter(damaged))
    damaged.remove(edge)
    assert not is_flat(graph, frozenset(damaged))


def test_clique_budget_shortfall_is_a_limit_not_a_false_proof(
    divergence_fixture,
) -> None:
    *_, composite = divergence_fixture
    assert clique_graph(composite.cover_graph, bound=25) is None
    limited = replace(composite, clique_bound=25)
    result = verify_triangular_cover_divergence_certificate(limited)
    assert result.status is CertificateVerificationStatus.LIMIT_REACHED


def test_lifted_cliques_and_induced_deck_action_match_the_handoff(
    divergence_fixture,
) -> None:
    _, _, cover, _, deck, clique_cover, composite = divergence_fixture
    inner = composite.theorem_certificate
    source_pair = clockwork_coaffine_pair(3, 0)

    assert (
        source_pair.graph.number_of_nodes(),
        source_pair.graph.number_of_edges(),
    ) == (18, 42)
    assert set(inner.map) == set(source_pair.graph)
    assert set(inner.map.values()) <= set(clique_cover)
    assert len({clique for clique in inner.map.values()}) == 18
    assert {Clique(clique) for clique in nx.find_cliques(cover)} == set(clique_cover)
    assert len(set(clique_cover)) == 26
    tau = inner.target_coaffination
    assert all(tau[tau[clique]] == clique for clique in clique_cover)
    distances = dict(nx.all_pairs_shortest_path_length(clique_cover))
    assert all(distances[clique][tau[clique]] >= 4 for clique in clique_cover)
    assert all(
        inner.map[source_pair.coaffination[vertex]] == tau[inner.map[vertex]]
        for vertex in source_pair.graph
    )
    assert all(
        inner.map[first] == inner.map[second]
        or clique_cover.has_edge(inner.map[first], inner.map[second])
        for first, second in source_pair.graph.edges()
    )
    assert all(
        nx.shortest_path_length(cover, vertex, deck[vertex]) >= 4 for vertex in cover
    )


def subgroup_voltage(cover: nx.Graph, graph: nx.Graph) -> tuple:
    return tuple(
        (
            first,
            second,
            next(
                sheet for sheet in (0, 1) if cover.has_edge((first, 0), (second, sheet))
            ),
        )
        for first, second in graph.edges()
    )


def test_voltage_provenance_replays_and_detects_damage(divergence_fixture) -> None:
    graph, _, cover, _, _, _, composite = divergence_fixture
    voltages = subgroup_voltage(cover, graph)
    with_voltage = replace(composite, voltage_cocycle=voltages)
    assert (
        verify_triangular_cover_divergence_certificate(with_voltage).status
        is CertificateVerificationStatus.VERIFIED
    )

    first, second, value = voltages[0]
    damaged = replace(
        composite, voltage_cocycle=((first, second, 1 - value), *voltages[1:])
    )
    assert (
        verify_triangular_cover_divergence_certificate(damaged).status
        is CertificateVerificationStatus.INVALID
    )


def test_nonflat_voltage_cover_is_not_a_triangular_cover(divergence_fixture) -> None:
    graph, _, _, _, _, _, composite = divergence_fixture
    damaged = set(VOLTAGE_ONE)
    damaged.remove(next(iter(damaged)))
    bad_cover = explicit_double_cover(graph, frozenset(damaged))
    changed = replace(
        composite,
        cover_graph=bad_cover,
        projection={vertex: vertex[0] for vertex in bad_cover},
        deck_transformation={
            vertex: (vertex[0], 1 - vertex[1]) for vertex in bad_cover
        },
        subgroup_words=(),
    )

    result = verify_triangular_cover_divergence_certificate(changed)
    assert result.status is CertificateVerificationStatus.INVALID


def test_certificate_for_a_different_base_graph_is_rejected(
    divergence_fixture,
) -> None:
    graph, _, _, _, _, _, composite = divergence_fixture
    other = graph.copy()
    other.remove_edge(*next(iter(other.edges())))
    changed = replace(composite, base_graph=other, subgroup_words=())

    result = verify_triangular_cover_divergence_certificate(changed)
    assert result.status is CertificateVerificationStatus.INVALID


def test_empty_base_graph_returns_invalid_certificate_result() -> None:
    certificate = TriangularCoverDivergenceCertificate(
        base_graph=nx.Graph(),
        cover_graph=nx.Graph(),
        projection={},
        deck_transformation={},
        theorem_certificate=Certificate(rule="theorem_2_6_clockwork_pair_map"),
    )

    result = verify_triangular_cover_divergence_certificate(certificate)

    assert result.status is CertificateVerificationStatus.INVALID
    assert "nonempty" in result.reason.lower()
