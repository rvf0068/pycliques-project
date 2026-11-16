"""Tests for explicitly bounded triangular-cover divergence discovery."""

from collections.abc import Hashable
from itertools import product
from random import Random

import networkx as nx
from pycliques.clockwork_search import CertificateVerificationStatus
from pycliques.coaffinations import CoaffinePair
from pycombtop.covering_divergence import (
    verify_triangular_cover_divergence_certificate,
)
from pycombtop.covering_divergence_search import (
    CoveringSearchStatus,
    _all_nonzero_cocycles,
    _bounded_pair_morphism,
    _triangles,
    search_covering_divergence,
)


def _is_flat(graph: nx.Graph, cocycle: tuple[tuple[object, object, int], ...]) -> bool:
    voltage = {frozenset((first, second)): value for first, second, value in cocycle}
    return all(
        voltage[frozenset((first, second))]
        ^ voltage[frozenset((second, third))]
        ^ voltage[frozenset((first, third))]
        == 0
        for first, second, third in _triangles(graph)
    )


def test_normalized_classes_are_flat_and_unique_on_cycle() -> None:
    graph = nx.cycle_graph(4)
    classes = _all_nonzero_cocycles(graph, max_classes=10)

    assert len(classes) == 1
    assert _is_flat(graph, classes[0])
    assert sum(value for _, _, value in classes[0]) % 2 == 1


def test_search_reports_complete_empty_cohomology_family() -> None:
    result = search_covering_divergence(
        nx.complete_graph(3),
        max_m=1,
        max_n=0,
    )

    assert result.status is CoveringSearchStatus.EXHAUSTED
    assert result.cover_classes_total == 0
    assert result.cover_classes_complete
    assert result.morphisms_complete
    assert result.certificate is None


def test_cover_class_cap_is_explicitly_incomplete() -> None:
    connected = nx.Graph()
    connected.add_edges_from(((0, 1), (1, 2), (2, 3), (3, 0)))
    connected.add_edges_from(((0, 4), (4, 5), (5, 6), (6, 7), (7, 0)))

    result = search_covering_divergence(
        connected,
        max_cover_classes=1,
        max_m=1,
        max_n=0,
    )

    assert result.status is CoveringSearchStatus.LIMIT_REACHED
    assert result.cover_classes_total == 3
    assert result.cover_classes_checked == 1
    assert not result.cover_classes_complete
    assert result.morphisms_complete


def test_morphism_budget_is_inconclusive_not_exhausted() -> None:
    graph = nx.from_graph6_bytes(b"IEjdnpy^O")
    result = search_covering_divergence(
        graph,
        max_cover_classes=4,
        max_source_order=18,
        max_m=3,
        max_n=0,
        max_morphism_states=0,
    )

    assert result.status is CoveringSearchStatus.LIMIT_REACHED
    assert result.cover_classes_total == 1
    assert result.cover_classes_complete
    assert not result.morphisms_complete
    assert result.certificate is None


def test_pair_morphism_search_handles_more_than_one_thousand_orbits() -> None:
    orbit_count = 1100
    source = nx.empty_graph(2 * orbit_count)
    sigma: dict[Hashable, Hashable] = {vertex: vertex ^ 1 for vertex in source}
    target = nx.empty_graph(2)
    tau: dict[Hashable, Hashable] = {0: 1, 1: 0}

    mapping, states, attempts, limited = _bounded_pair_morphism(
        CoaffinePair(source, sigma),
        CoaffinePair(target, tau),
        max_states=orbit_count + 2,
        max_assignments=orbit_count + 2,
    )

    assert not limited
    assert mapping is not None
    assert set(mapping) == set(source)
    assert states <= orbit_count + 2
    assert attempts <= orbit_count + 2
    assert all(mapping[sigma[vertex]] == tau[mapping[vertex]] for vertex in source)


def test_pair_morphism_search_respects_each_budget_and_zero_budgets() -> None:
    orbit_count = 1100
    source = nx.empty_graph(2 * orbit_count)
    sigma: dict[Hashable, Hashable] = {vertex: vertex ^ 1 for vertex in source}
    target = nx.empty_graph(2)
    tau: dict[Hashable, Hashable] = {0: 1, 1: 0}
    source_pair = CoaffinePair(source, sigma)
    target_pair = CoaffinePair(target, tau)

    _, states, _, limited = _bounded_pair_morphism(
        source_pair,
        target_pair,
        max_states=5,
        max_assignments=10_000,
    )
    assert limited
    assert states <= 5

    _, _, attempts, limited = _bounded_pair_morphism(
        source_pair,
        target_pair,
        max_states=10_000,
        max_assignments=7,
    )
    assert limited
    assert attempts <= 7

    for state_budget, assignment_budget in ((0, 10), (10, 0)):
        _, states, attempts, limited = _bounded_pair_morphism(
            source_pair,
            target_pair,
            max_states=state_budget,
            max_assignments=assignment_budget,
        )
        assert limited
        assert states <= state_budget
        assert attempts <= assignment_budget


def test_pair_morphism_search_exhausts_unsatisfiable_instance() -> None:
    source = nx.Graph([(0, 1)])
    target = nx.empty_graph(2)
    sigma: dict[Hashable, Hashable] = {0: 1, 1: 0}
    tau: dict[Hashable, Hashable] = {0: 1, 1: 0}

    mapping, states, attempts, limited = _bounded_pair_morphism(
        CoaffinePair(source, sigma),
        CoaffinePair(target, tau),
        max_states=10,
        max_assignments=10,
    )

    assert mapping is None
    assert states == 1
    # Unary orbit constraint: the source mates are adjacent but the
    # target mates are not. AC-3 proves impossibility without guessing.
    assert attempts == 0
    assert not limited


def test_cover_and_clique_construction_budgets_are_reported() -> None:
    graph = nx.cycle_graph(4)
    cover_limited = search_covering_divergence(
        graph,
        max_cover_vertices=7,
        max_m=2,
        max_n=0,
    )
    clique_limited = search_covering_divergence(
        graph,
        max_clique_edges=0,
        max_source_order=12,
        max_m=2,
        max_n=0,
    )

    assert cover_limited.status is CoveringSearchStatus.LIMIT_REACHED
    assert not cover_limited.morphisms_complete
    assert any("cover construction" in reason for reason in cover_limited.reasons)
    assert clique_limited.status is CoveringSearchStatus.LIMIT_REACHED
    assert not clique_limited.morphisms_complete
    assert any(
        "clique-graph construction" in reason for reason in clique_limited.reasons
    )


def test_search_finds_and_replays_the_known_covering_certificate() -> None:
    graph = nx.from_graph6_bytes(b"IEjdnpy^O")
    result = search_covering_divergence(
        graph,
        max_cover_classes=4,
        max_cover_vertices=40,
        max_cover_edges=100,
        max_clique_vertices=30,
        max_clique_edges=200,
        max_source_order=18,
        max_m=3,
        max_n=0,
        max_morphism_states=100_000,
        max_morphism_assignments=1_000_000,
    )

    assert result.status is CoveringSearchStatus.FOUND
    assert result.certificate is not None
    assert result.cover_classes_total == 1
    assert result.cover_classes_complete
    assert not result.morphisms_complete
    assert result.morphism_states > 0
    assert result.morphism_assignments > 0
    assert (
        verify_triangular_cover_divergence_certificate(result.certificate).status
        is CertificateVerificationStatus.VERIFIED
    )
    damaged_voltage = list(result.certificate.voltage_cocycle)
    first, second, value = damaged_voltage[0]
    damaged_voltage[0] = (first, second, 1 - value)
    damaged_certificate = type(result.certificate)(
        **{
            **result.certificate.__dict__,
            "voltage_cocycle": tuple(damaged_voltage),
        }
    )
    assert (
        verify_triangular_cover_divergence_certificate(damaged_certificate).status
        is CertificateVerificationStatus.INVALID
    )


def _brute_force_pair_morphism(
    source_pair: CoaffinePair, target_pair: CoaffinePair
) -> dict[Hashable, Hashable] | None:
    """Independent exhaustive reference for small involutive pairs."""
    source = source_pair.graph
    sigma = source_pair.coaffination
    target = target_pair.graph
    tau = target_pair.coaffination
    seen: set[Hashable] = set()
    orbits: list[tuple[Hashable, ...]] = []
    for vertex in source:
        if vertex in seen:
            continue
        partner = sigma[vertex]
        orbit = (vertex,) if partner == vertex else (vertex, partner)
        orbits.append(orbit)
        seen.update(orbit)
    for choice in product(tuple(target), repeat=len(orbits)):
        values = {
            member: (value if pos == 0 else tau[value])
            for orbit, value in zip(orbits, choice)
            for pos, member in enumerate(orbit)
        }
        if any(values[sigma[u]] != tau[values[u]] for u in source):
            continue
        if all(
            values[u] == values[v] or target.has_edge(values[u], values[v])
            for u, v in source.edges()
        ):
            return values
    return None


def _pair_from_involutive_graph(
    labels: tuple[Hashable, ...], seed: int
) -> CoaffinePair:
    """Generate small graph automorphisms with deterministic fixed/mate orbits."""
    rng = Random(seed)
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    n = len(labels)
    sigma = {labels[i]: labels[i ^ 1] for i in range(n - n % 2)}
    if n % 2:
        sigma[labels[-1]] = labels[-1]
    # Close each random edge orbit under the involution, so sigma
    # really is a graph automorphism (including fixed vertices).
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.34:
                u, v = labels[i], labels[j]
                graph.add_edge(u, v)
                if sigma[u] != sigma[v]:
                    graph.add_edge(sigma[u], sigma[v])
    return CoaffinePair(graph, sigma)


def test_pair_morphism_matches_independent_exhaustive_reference() -> None:
    """AC-3/MRV must neither lose maps nor invent graphs morphisms."""
    for seed in range(80):
        # Alternate between fixed points and 2-cycles, with both integer
        # and nonnumeric labels in the source and target.
        source_labels = tuple(f"s{j}" for j in range(4 + seed % 3))
        target_labels = tuple(f"t{j}" for j in range(3 + seed % 4))
        source = _pair_from_involutive_graph(source_labels, seed)
        target = _pair_from_involutive_graph(target_labels, seed + 1000)
        expected = _brute_force_pair_morphism(source, target)
        found, states, attempts, limited = _bounded_pair_morphism(
            source, target, max_states=200_000, max_assignments=200_000
        )
        assert not limited, seed
        assert (found is not None) == (expected is not None), seed
        assert states >= 1
        assert 0 <= attempts <= 200_000
        if found is not None:
            assert set(found) == set(source.graph)
            assert all(
                found[source.coaffination[u]] == target.coaffination[found[u]]
                for u in source.graph
            )
            assert all(
                found[u] == found[v] or target.graph.has_edge(found[u], found[v])
                for u, v in source.graph.edges()
            )


def test_pair_morphism_prunes_inconsistent_cross_orbit_edges() -> None:
    # Two source 2-cycles linked by a square's edges, mapping into an
    # edgeless two-vertex target swapped by tau. Every possible orbit
    # choice violates at least one edge, so AC-3 can prove unsat early.
    source = nx.cycle_graph(4)
    sigma = {0: 2, 2: 0, 1: 3, 3: 1}
    target = nx.empty_graph(2)
    tau = {0: 1, 1: 0}
    mapping, states, attempts, limited = _bounded_pair_morphism(
        CoaffinePair(source, sigma),
        CoaffinePair(target, tau),
        max_states=10,
        max_assignments=10,
    )
    assert mapping is None
    assert not limited
    assert states == 1
    assert attempts == 0


def test_pair_morphism_allows_adjacent_vertices_to_map_equal() -> None:
    source = nx.Graph([(0, 1)])
    target = nx.empty_graph(1)
    mapping, states, attempts, limited = _bounded_pair_morphism(
        CoaffinePair(source, {0: 1, 1: 0}),
        CoaffinePair(target, {0: 0}),
        max_states=10,
        max_assignments=10,
    )
    assert mapping == {0: 0, 1: 0}
    assert not limited
    assert states == 2
    assert attempts == 1


def test_pair_morphism_respects_tight_budgets_after_propagation() -> None:
    source = nx.empty_graph(4)
    target = nx.empty_graph(2)
    sigma = {0: 1, 1: 0, 2: 3, 3: 2}
    tau = {0: 1, 1: 0}
    source_pair = CoaffinePair(source, sigma)
    target_pair = CoaffinePair(target, tau)
    for max_states, max_assignments in ((0, 10), (1, 10), (10, 0), (10, 1)):
        result, states, attempts, limited = _bounded_pair_morphism(
            source_pair,
            target_pair,
            max_states=max_states,
            max_assignments=max_assignments,
        )
        assert result is None
        assert limited
        assert states <= max_states
        assert attempts <= max_assignments


def test_pair_morphism_rejects_non_involutive_pair_maps() -> None:
    # The bounded double-cover search is specifically for involutive
    # coaffinations. Formerly longer orbits were silently omitted.
    source = nx.empty_graph(3)
    target = nx.empty_graph(2)
    source_pair = CoaffinePair(source, {0: 1, 1: 2, 2: 0})
    target_pair = CoaffinePair(target, {0: 1, 1: 0})
    import pytest

    with pytest.raises(ValueError, match="involution"):
        _bounded_pair_morphism(
            source_pair,
            target_pair,
            max_states=100,
            max_assignments=100,
        )
