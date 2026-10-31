"""Bounded induced-clockwork discovery and divergence-certificate searches."""

from __future__ import annotations

import itertools
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum, auto
from math import comb

import networkx as nx

from .clockwork import (
    ClockworkStructure,
    is_clique_divergent_clockwork,
    recognize_clockwork,
    verify_clockwork_structure,
)
from .retractions import is_map, retracts


class ClockworkSearchStatus(Enum):
    """Completion state for a bounded clockwork search."""

    FOUND = auto()
    EXHAUSTED = auto()
    LIMIT_REACHED = auto()


@dataclass(frozen=True)
class InducedClockworkWitness:
    """A verified induced clockwork subgraph and its structural evidence."""

    vertices: frozenset[Hashable]
    graph: nx.Graph
    structure: ClockworkStructure
    divergent: bool | None
    behavior_reason: str
    iterate_index: int = 0


@dataclass(frozen=True)
class ClockworkSearchResult:
    """Results from bounded induced-clockwork discovery.

    ``EXHAUSTED`` means every induced candidate up to ``max_order`` was
    checked. ``LIMIT_REACHED`` means candidate enumeration stopped early;
    any witnesses already returned remain valid.
    """

    status: ClockworkSearchStatus
    witnesses: tuple[InducedClockworkWitness, ...]
    candidates_checked: int
    candidates_total: int
    reason: str


@dataclass(frozen=True)
class ClockworkRetractionCertificate:
    """Verified retraction from a graph onto an induced divergent clockwork."""

    source_graph: nx.Graph
    target_graph: nx.Graph
    target_vertices: frozenset[Hashable]
    retraction_map: dict[Hashable, Hashable]
    structure: ClockworkStructure
    iterate_index: int = 0


@dataclass(frozen=True)
class ClockworkRetractionSearchResult:
    """Result of bounded search for a divergent clockwork retract."""

    status: ClockworkSearchStatus
    certificate: ClockworkRetractionCertificate | None
    candidates_checked: int
    retraction_attempts: int
    reason: str


def _validate_search_limits(
    max_order: int, max_candidates: int, max_results: int
) -> None:
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")


def find_induced_clockwork_subgraphs(
    graph: nx.Graph,
    *,
    max_order: int = 12,
    max_candidates: int = 5000,
    max_results: int = 1,
    divergent_only: bool = True,
    iterate_index: int = 0,
) -> ClockworkSearchResult:
    """Find verified induced clockwork subgraphs up to a bounded order.

    Candidates are checked in increasing order and must contain at least
    nine vertices (three core segments and three crown segments of size at
    least two). ``max_candidates`` counts vertex subsets checked. A found
    induced divergent clockwork is structural evidence only; it does not
    certify divergence of the containing graph without a retraction.

    Parameters
    ----------
    graph : networkx.Graph
        Host graph; its vertex labels are preserved in returned witnesses.
    max_order : int, default 12
        Largest induced-subgraph order enumerated.
    max_candidates : int, default 5000
        Maximum number of vertex subsets to inspect.
    max_results : int, default 1
        Stop after this many witnesses.
    divergent_only : bool, default True
        Return only structures whose clockwork theorem certifies divergence.
    iterate_index : int, default 0
        Index of this graph in an externally computed clique sequence.

    Returns
    -------
    ClockworkSearchResult
        Status, witnesses, checked/total candidate counts, and a reason.
    """
    _validate_search_limits(max_order, max_candidates, max_results)
    if iterate_index < 0:
        raise ValueError("iterate_index must be non-negative")

    upper_order = min(max_order, graph.number_of_nodes())
    candidate_orders = range(9, upper_order + 1)
    candidates_total = sum(
        comb(graph.number_of_nodes(), order) for order in candidate_orders
    )
    witnesses: list[InducedClockworkWitness] = []
    checked = 0

    for order in candidate_orders:
        for vertices in itertools.combinations(graph.nodes(), order):
            if checked >= max_candidates:
                return ClockworkSearchResult(
                    ClockworkSearchStatus.LIMIT_REACHED,
                    tuple(witnesses),
                    checked,
                    candidates_total,
                    f"Stopped after the candidate limit ({max_candidates}).",
                )
            checked += 1
            candidate = graph.subgraph(vertices).copy()
            if not nx.is_connected(candidate):
                continue
            recognized, structure_or_reason = recognize_clockwork(
                candidate, fallback_max_order=min(max_order, 12)
            )
            if not recognized:
                continue
            assert isinstance(structure_or_reason, ClockworkStructure)
            divergent, behavior_reason = is_clique_divergent_clockwork(candidate)
            if divergent_only and divergent is not True:
                continue
            witnesses.append(
                InducedClockworkWitness(
                    vertices=frozenset(vertices),
                    graph=candidate,
                    structure=structure_or_reason,
                    divergent=divergent,
                    behavior_reason=behavior_reason,
                    iterate_index=iterate_index,
                )
            )
            if len(witnesses) >= max_results:
                status = (
                    ClockworkSearchStatus.EXHAUSTED
                    if checked == candidates_total
                    else ClockworkSearchStatus.FOUND
                )
                return ClockworkSearchResult(
                    status,
                    tuple(witnesses),
                    checked,
                    candidates_total,
                    "Verified clockwork witness found.",
                )

    return ClockworkSearchResult(
        ClockworkSearchStatus.FOUND if witnesses else ClockworkSearchStatus.EXHAUSTED,
        tuple(witnesses),
        checked,
        candidates_total,
        "Verified clockwork witness found."
        if witnesses
        else "All candidates within the specified order bound were checked.",
    )


def _map_structure(
    structure: ClockworkStructure,
    graph: nx.Graph,
    mapping: dict[Hashable, Hashable],
) -> ClockworkStructure | None:
    """Transport and re-verify a structure through an induced embedding."""
    B_segments = [
        {mapping[vertex] for vertex in segment} for segment in structure.B_segments
    ]
    C_segments = [
        {mapping[vertex] for vertex in segment} for segment in structure.C_segments
    ]
    core_orders = [
        [mapping[vertex] for vertex in order] for order in structure.core_orders
    ]
    valid, result = verify_clockwork_structure(
        graph, B_segments, C_segments, core_orders
    )
    if not valid:
        return None
    assert isinstance(result, ClockworkStructure)
    return result


def find_divergent_clockwork_retract(
    graph: nx.Graph,
    *,
    max_target_order: int = 12,
    max_candidates: int = 5000,
    max_retraction_attempts: int = 20,
    iterate_index: int = 0,
) -> ClockworkRetractionSearchResult:
    """Search for a verified retraction onto an induced divergent clockwork.

    Candidate subsets and retraction targets are bounded independently.
    ``max_retraction_attempts`` counts divergent clockwork targets passed to
    the existing retraction search; an exhausted or interrupted search does
    not imply convergence. The returned map is explicitly re-verified.

    Parameters
    ----------
    graph : networkx.Graph
        Graph searched as the source of a retraction.
    max_target_order : int, default 12
        Largest induced clockwork target order considered.
    max_candidates : int, default 5000
        Maximum number of vertex subsets to inspect.
    max_retraction_attempts : int, default 20
        Maximum number of divergent clockwork targets tested.
    iterate_index : int, default 0
        Index of this graph in an externally computed clique sequence.

    Returns
    -------
    ClockworkRetractionSearchResult
        Search status and, on success, a verified divergence certificate.
    """
    if max_target_order < 0:
        raise ValueError("max_target_order must be non-negative")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if max_retraction_attempts < 1:
        raise ValueError("max_retraction_attempts must be at least 1")
    if iterate_index < 0:
        raise ValueError("iterate_index must be non-negative")

    upper_order = min(max_target_order, graph.number_of_nodes())
    candidate_orders = range(9, upper_order + 1)
    candidates_total = sum(
        comb(graph.number_of_nodes(), order) for order in candidate_orders
    )
    checked = 0
    attempts = 0

    for order in candidate_orders:
        for vertices in itertools.combinations(graph.nodes(), order):
            if checked >= max_candidates:
                return ClockworkRetractionSearchResult(
                    ClockworkSearchStatus.LIMIT_REACHED,
                    None,
                    checked,
                    attempts,
                    f"Stopped after the candidate limit ({max_candidates}).",
                )
            checked += 1
            candidate = graph.subgraph(vertices).copy()
            if not nx.is_connected(candidate):
                continue
            recognized, structure_or_reason = recognize_clockwork(
                candidate, fallback_max_order=min(max_target_order, 12)
            )
            if not recognized:
                continue
            assert isinstance(structure_or_reason, ClockworkStructure)
            divergent, _reason = is_clique_divergent_clockwork(candidate)
            if divergent is not True:
                continue
            if attempts >= max_retraction_attempts:
                return ClockworkRetractionSearchResult(
                    ClockworkSearchStatus.LIMIT_REACHED,
                    None,
                    checked,
                    attempts,
                    "Stopped after the retraction-attempt limit "
                    f"({max_retraction_attempts}).",
                )
            attempts += 1
            result = retracts(graph, candidate)
            if not isinstance(result, tuple):
                continue
            retraction_map, inclusion_map = result
            image_nodes = set(inclusion_map.values())
            image_graph = graph.subgraph(image_nodes).copy()
            target_mapping = {
                vertex: inclusion_map[target]
                for vertex, target in retraction_map.items()
            }
            if (
                set(target_mapping) != set(graph)
                or not is_map(graph, image_graph, target_mapping)
                or any(target_mapping[vertex] != vertex for vertex in image_nodes)
            ):
                continue
            image_structure = _map_structure(
                structure_or_reason, image_graph, inclusion_map
            )
            if image_structure is None:
                continue
            image_divergent, _image_reason = is_clique_divergent_clockwork(image_graph)
            if image_divergent is not True:
                continue
            certificate = ClockworkRetractionCertificate(
                source_graph=graph,
                target_graph=image_graph,
                target_vertices=frozenset(image_nodes),
                retraction_map=target_mapping,
                structure=image_structure,
                iterate_index=iterate_index,
            )
            return ClockworkRetractionSearchResult(
                ClockworkSearchStatus.FOUND,
                certificate,
                checked,
                attempts,
                "Verified retraction onto an induced divergent clockwork.",
            )

    status = (
        ClockworkSearchStatus.LIMIT_REACHED
        if checked < candidates_total
        else ClockworkSearchStatus.EXHAUSTED
    )
    return ClockworkRetractionSearchResult(
        status,
        None,
        checked,
        attempts,
        "No divergent clockwork retract was found within the search bounds.",
    )
