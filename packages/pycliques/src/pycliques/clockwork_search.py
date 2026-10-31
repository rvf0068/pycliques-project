"""Bounded induced-clockwork discovery and divergence-certificate searches."""

from __future__ import annotations

import itertools
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum, auto
from math import comb, inf

import networkx as nx

from .cliques import clique_graph
from .clockwork import (
    ClockworkDivergenceStatus,
    ClockworkRecognitionStatus,
    ClockworkStructure,
    clockwork_divergence_result,
    recognize_clockwork_detailed,
    verify_clockwork_structure,
)
from .dominated import completely_pared_graph
from .retractions import (
    RetractionSearchStatus,
    bounded_retraction,
    is_map,
)


class ClockworkSearchStatus(Enum):
    """Completion state for a bounded clockwork search."""

    FOUND = auto()
    EXHAUSTED = auto()
    LIMIT_REACHED = auto()


class CertificateVerificationStatus(Enum):
    """Outcome of independently checking a clockwork retraction certificate."""

    VERIFIED = auto()
    INVALID = auto()
    LIMIT_REACHED = auto()


@dataclass(frozen=True)
class CertificateVerificationResult:
    """Detailed outcome of certificate verification."""

    status: CertificateVerificationStatus
    reason: str


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
    checked conclusively. ``LIMIT_REACHED`` means candidate enumeration,
    recognition, or divergence analysis was incomplete; any witnesses
    already returned remain valid.
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
    origin_graph: nx.Graph | None = None
    iterate_graphs: tuple[nx.Graph, ...] = ()
    recognition_max_order: int = 12
    recognition_max_candidates: int = 5000


@dataclass(frozen=True)
class ClockworkRetractionSearchResult:
    """Result of bounded search for a divergent clockwork retract."""

    status: ClockworkSearchStatus
    certificate: ClockworkRetractionCertificate | None
    candidates_checked: int
    retraction_attempts: int
    reason: str
    states_explored: int = 0
    assignments_tried: int = 0


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
    recognition_max_order: int = 12,
    recognition_max_candidates: int = 5000,
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
    if recognition_max_order < 0:
        raise ValueError("recognition_max_order must be non-negative")
    if recognition_max_candidates < 1:
        raise ValueError("recognition_max_candidates must be at least 1")

    upper_order = min(max_order, graph.number_of_nodes())
    candidate_orders = range(9, upper_order + 1)
    candidates_total = sum(
        comb(graph.number_of_nodes(), order) for order in candidate_orders
    )
    witnesses: list[InducedClockworkWitness] = []
    checked = 0
    recognition_incomplete = False

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
            recognition = recognize_clockwork_detailed(
                candidate,
                fallback_max_order=recognition_max_order,
                fallback_max_candidates=recognition_max_candidates,
            )
            if recognition.status is ClockworkRecognitionStatus.LIMIT_REACHED:
                recognition_incomplete = True
                continue
            if recognition.status is ClockworkRecognitionStatus.NOT_FOUND:
                continue
            if recognition.structure is None:
                raise RuntimeError("Successful clockwork recognition has no structure.")
            divergence = clockwork_divergence_result(
                candidate,
                structure=recognition.structure,
                fallback_max_order=recognition_max_order,
                fallback_max_candidates=recognition_max_candidates,
            )
            if divergence.status is ClockworkDivergenceStatus.LIMIT_REACHED:
                recognition_incomplete = True
                continue
            if divergence.status is ClockworkDivergenceStatus.NOT_CLOCKWORK:
                recognition_incomplete = True
                continue
            divergent = divergence.status is ClockworkDivergenceStatus.DIVERGENT
            if divergent_only and not divergent:
                continue
            witnesses.append(
                InducedClockworkWitness(
                    vertices=frozenset(vertices),
                    graph=candidate,
                    structure=recognition.structure,
                    divergent=divergent,
                    behavior_reason=divergence.reason,
                    iterate_index=iterate_index,
                )
            )
            if len(witnesses) >= max_results:
                if checked < candidates_total:
                    status = ClockworkSearchStatus.FOUND
                    reason = "Verified clockwork witness found."
                elif recognition_incomplete:
                    status = ClockworkSearchStatus.LIMIT_REACHED
                    reason = (
                        "A verified clockwork witness was found, but recognition "
                        "or divergence analysis was inconclusive for another candidate."
                    )
                else:
                    status = ClockworkSearchStatus.EXHAUSTED
                    reason = (
                        "Verified clockwork witness found; all candidates were checked."
                    )
                return ClockworkSearchResult(
                    status,
                    tuple(witnesses),
                    checked,
                    candidates_total,
                    reason,
                )

    status = (
        ClockworkSearchStatus.LIMIT_REACHED
        if recognition_incomplete
        else ClockworkSearchStatus.FOUND
        if witnesses
        else ClockworkSearchStatus.EXHAUSTED
    )
    return ClockworkSearchResult(
        status,
        tuple(witnesses),
        checked,
        candidates_total,
        (
            "All candidate subsets were checked, but recognition or "
            "divergence analysis was inconclusive for at least one candidate."
            if recognition_incomplete
            else "Verified clockwork witness found."
            if witnesses
            else "All candidates within the specified order bound were checked."
        ),
    )


def _same_graph(left: nx.Graph, right: nx.Graph) -> bool:
    return set(left) == set(right) and {frozenset(edge) for edge in left.edges()} == {
        frozenset(edge) for edge in right.edges()
    }


def verify_clockwork_retraction_certificate(
    certificate: ClockworkRetractionCertificate,
) -> bool:
    """Return whether ``certificate`` passes unrestricted independent replay."""
    result = verify_clockwork_retraction_certificate_detailed(certificate)
    return result.status is CertificateVerificationStatus.VERIFIED


def verify_clockwork_retraction_certificate_detailed(
    certificate: ClockworkRetractionCertificate,
    *,
    clique_bound: int | float | None = None,
    max_edges: int | None = None,
) -> CertificateVerificationResult:
    """Independently verify a retraction and its clique-iterate provenance.

    Positive-index certificates replay every transition. ``clique_bound``
    and ``max_edges`` limit each reconstructed clique graph; a reconstruction
    stopped by either limit returns ``LIMIT_REACHED`` rather than ``INVALID``.
    ``None`` for ``clique_bound`` means no clique-count limit.
    """
    if clique_bound is not None and clique_bound < 0:
        raise ValueError("clique_bound must be non-negative or None")
    if max_edges is not None and max_edges < 0:
        raise ValueError("max_edges must be non-negative or None")

    def invalid(reason: str) -> CertificateVerificationResult:
        return CertificateVerificationResult(
            CertificateVerificationStatus.INVALID, reason
        )

    source = certificate.source_graph
    target = certificate.target_graph
    if (
        certificate.iterate_index < 0
        or certificate.recognition_max_order < 0
        or certificate.recognition_max_candidates < 1
    ):
        return invalid("Certificate contains invalid indices or recognition bounds.")
    target_nodes = set(certificate.target_vertices)
    if target_nodes != set(target) or not target_nodes <= set(source):
        return invalid("Target vertices do not describe an induced source subgraph.")
    induced_target = source.subgraph(target_nodes)
    if not _same_graph(induced_target, target):
        return invalid("Certificate target is not the induced source subgraph.")

    mapping = certificate.retraction_map
    if set(mapping) != set(source) or not set(mapping.values()) <= target_nodes:
        return invalid("Retraction map is not total or has images outside the target.")
    if not is_map(source, target, mapping):
        return invalid("Retraction map does not preserve source edges.")
    if any(mapping[vertex] != vertex for vertex in target_nodes):
        return invalid("Retraction map does not fix target vertices pointwise.")

    valid, verified_structure = verify_clockwork_structure(
        target,
        certificate.structure.B_segments,
        certificate.structure.C_segments,
        certificate.structure.core_orders,
    )
    if not valid or not isinstance(verified_structure, ClockworkStructure):
        return invalid("Certificate clockwork structure is invalid.")
    divergence = clockwork_divergence_result(
        target,
        structure=verified_structure,
        fallback_max_order=certificate.recognition_max_order,
        fallback_max_candidates=certificate.recognition_max_candidates,
    )
    if divergence.status is ClockworkDivergenceStatus.LIMIT_REACHED:
        return CertificateVerificationResult(
            CertificateVerificationStatus.LIMIT_REACHED,
            "Clockwork divergence verification reached a recognition limit.",
        )
    if divergence.status is not ClockworkDivergenceStatus.DIVERGENT:
        return invalid("Certificate target is not proven clique-divergent.")

    if certificate.iterate_index == 0:
        if certificate.iterate_graphs:
            if len(certificate.iterate_graphs) != 1 or not _same_graph(
                certificate.iterate_graphs[0], source
            ):
                return invalid(
                    "Iterate-zero prefix must contain only the source graph."
                )
        if certificate.origin_graph is not None and not _same_graph(
            certificate.origin_graph, source
        ):
            return invalid("Iterate-zero origin graph differs from the source.")
        return CertificateVerificationResult(
            CertificateVerificationStatus.VERIFIED,
            "Retraction certificate verified at iterate zero.",
        )

    iterates = certificate.iterate_graphs
    if (
        certificate.origin_graph is None
        or len(iterates) != certificate.iterate_index + 1
        or not _same_graph(iterates[0], certificate.origin_graph)
        or not _same_graph(iterates[-1], source)
    ):
        return invalid("Clique-iterate provenance has an invalid prefix shape.")
    replay_bound = inf if clique_bound is None else clique_bound
    for previous, current in itertools.pairwise(iterates):
        next_graph = clique_graph(previous, replay_bound, max_edges=max_edges)
        if next_graph is None:
            return CertificateVerificationResult(
                CertificateVerificationStatus.LIMIT_REACHED,
                "Clique-graph replay exceeded its configured resource bound.",
            )
        if not _same_graph(completely_pared_graph(next_graph), current):
            return invalid(
                "A stored iterate does not match its clique-graph transition."
            )
    return CertificateVerificationResult(
        CertificateVerificationStatus.VERIFIED,
        "Retraction certificate and clique-iterate provenance verified.",
    )


def _validate_iterate_provenance(
    graph: nx.Graph,
    iterate_index: int,
    origin_graph: nx.Graph | None,
    iterate_graphs: tuple[nx.Graph, ...],
) -> str | None:
    """Return a validation error for the supplied labelled iterate prefix."""
    if iterate_index < 0:
        return "iterate_index must be non-negative"
    if origin_graph is not None and not isinstance(origin_graph, nx.Graph):
        return "origin_graph must be a NetworkX graph"
    if not all(isinstance(iterate, nx.Graph) for iterate in iterate_graphs):
        return "every iterate_graphs entry must be a NetworkX graph"
    if iterate_index > 0 and (origin_graph is None or not iterate_graphs):
        return (
            "positive iterate_index requires origin_graph and the complete "
            "iterate_graphs provenance prefix"
        )
    if iterate_graphs and len(iterate_graphs) != iterate_index + 1:
        return "iterate_graphs length must equal iterate_index + 1"
    if (
        origin_graph is not None
        and iterate_graphs
        and not _same_graph(iterate_graphs[0], origin_graph)
    ):
        return "iterate_graphs[0] must equal origin_graph"
    if iterate_graphs and not _same_graph(iterate_graphs[-1], graph):
        return "the final iterate_graphs entry must equal graph"
    if (
        iterate_index == 0
        and origin_graph is not None
        and not _same_graph(origin_graph, graph)
    ):
        return "at iterate zero, origin_graph must equal graph"
    return None


def find_divergent_clockwork_retract(
    graph: nx.Graph,
    *,
    max_target_order: int = 12,
    max_candidates: int = 5000,
    max_retraction_attempts: int = 20,
    max_retraction_states: int = 100_000,
    max_retraction_assignments: int = 1_000_000,
    iterate_index: int = 0,
    origin_graph: nx.Graph | None = None,
    iterate_graphs: tuple[nx.Graph, ...] = (),
    recognition_max_order: int = 12,
    recognition_max_candidates: int = 5000,
    replay_clique_bound: int | float | None = None,
    replay_max_edges: int | None = None,
) -> ClockworkRetractionSearchResult:
    """Search for a verified retraction onto an induced divergent clockwork.

    Candidate subsets and retraction targets are bounded independently.
    ``max_retraction_attempts`` counts divergent clockwork targets tested.
    ``max_retraction_states`` and ``max_retraction_assignments`` also bound
    each target's internal retraction backtracking, without enumerating
    automorphisms. An exhausted or interrupted search does not imply
    convergence. The returned map is explicitly re-verified.

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
    max_retraction_states : int, default 100000
        Maximum backtracking states allowed per target.
    max_retraction_assignments : int, default 1000000
        Maximum vertex-to-target assignments tested per target.
    iterate_index : int, default 0
        Index of this graph in an externally computed clique sequence.
    origin_graph, iterate_graphs
        Optional provenance for verifying that this is the specified clique
        iterate of ``origin_graph``. Supply the complete cached prefix.
    recognition_max_order, recognition_max_candidates
        Bounds passed to clockwork recognition and dominated reduction.

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
    if max_retraction_states < 1:
        raise ValueError("max_retraction_states must be at least 1")
    if max_retraction_assignments < 1:
        raise ValueError("max_retraction_assignments must be at least 1")
    if iterate_index < 0:
        raise ValueError("iterate_index must be non-negative")
    if recognition_max_order < 0:
        raise ValueError("recognition_max_order must be non-negative")
    if recognition_max_candidates < 1:
        raise ValueError("recognition_max_candidates must be at least 1")
    if replay_clique_bound is not None and replay_clique_bound < 0:
        raise ValueError("replay_clique_bound must be non-negative or None")
    if replay_max_edges is not None and replay_max_edges < 0:
        raise ValueError("replay_max_edges must be non-negative or None")
    provenance_error = _validate_iterate_provenance(
        graph, iterate_index, origin_graph, iterate_graphs
    )
    if provenance_error is not None:
        raise ValueError(provenance_error)

    upper_order = min(max_target_order, graph.number_of_nodes())
    candidate_orders = range(9, upper_order + 1)
    candidates_total = sum(
        comb(graph.number_of_nodes(), order) for order in candidate_orders
    )
    checked = 0
    attempts = 0
    states_explored = 0
    assignments_tried = 0
    recognition_incomplete = False
    retraction_incomplete = False

    for order in candidate_orders:
        for vertices in itertools.combinations(graph.nodes(), order):
            if checked >= max_candidates:
                return ClockworkRetractionSearchResult(
                    ClockworkSearchStatus.LIMIT_REACHED,
                    None,
                    checked,
                    attempts,
                    f"Stopped after the candidate limit ({max_candidates}).",
                    states_explored,
                    assignments_tried,
                )
            checked += 1
            candidate = graph.subgraph(vertices).copy()
            if not nx.is_connected(candidate):
                continue
            recognition = recognize_clockwork_detailed(
                candidate,
                fallback_max_order=recognition_max_order,
                fallback_max_candidates=recognition_max_candidates,
            )
            if recognition.status is ClockworkRecognitionStatus.LIMIT_REACHED:
                recognition_incomplete = True
                continue
            if recognition.status is ClockworkRecognitionStatus.NOT_FOUND:
                continue
            if recognition.structure is None:
                raise RuntimeError("Successful clockwork recognition has no structure.")
            divergence = clockwork_divergence_result(
                candidate,
                structure=recognition.structure,
                fallback_max_order=recognition_max_order,
                fallback_max_candidates=recognition_max_candidates,
            )
            if divergence.status is ClockworkDivergenceStatus.LIMIT_REACHED:
                recognition_incomplete = True
                continue
            if divergence.status is ClockworkDivergenceStatus.NOT_CLOCKWORK:
                recognition_incomplete = True
                continue
            if divergence.status is not ClockworkDivergenceStatus.DIVERGENT:
                continue
            if attempts >= max_retraction_attempts:
                return ClockworkRetractionSearchResult(
                    ClockworkSearchStatus.LIMIT_REACHED,
                    None,
                    checked,
                    attempts,
                    "Stopped after the retraction-attempt limit "
                    f"({max_retraction_attempts}).",
                    states_explored,
                    assignments_tried,
                )
            attempts += 1
            result = bounded_retraction(
                graph,
                candidate,
                max_states=max_retraction_states,
                max_assignments=max_retraction_assignments,
            )
            states_explored += result.states_explored
            assignments_tried += result.assignments_tried
            if result.status is RetractionSearchStatus.LIMIT_REACHED:
                retraction_incomplete = True
                continue
            if result.status is RetractionSearchStatus.EXHAUSTED:
                continue
            if result.retraction is None or result.inclusion is None:
                raise RuntimeError("Successful bounded retraction has no mapping.")
            retraction_map, inclusion_map = result.retraction, result.inclusion
            image_nodes = set(inclusion_map.values())
            image_graph = graph.subgraph(image_nodes).copy()
            target_mapping = retraction_map
            if (
                set(target_mapping) != set(graph)
                or not is_map(graph, image_graph, target_mapping)
                or any(target_mapping[vertex] != vertex for vertex in image_nodes)
            ):
                raise RuntimeError("Bounded retraction returned an invalid map.")
            image_structure = recognition.structure
            certificate = ClockworkRetractionCertificate(
                source_graph=graph,
                target_graph=image_graph,
                target_vertices=frozenset(image_nodes),
                retraction_map=target_mapping,
                structure=image_structure,
                iterate_index=iterate_index,
                origin_graph=origin_graph,
                iterate_graphs=iterate_graphs,
                recognition_max_order=recognition_max_order,
                recognition_max_candidates=recognition_max_candidates,
            )
            verification = verify_clockwork_retraction_certificate_detailed(
                certificate,
                clique_bound=replay_clique_bound,
                max_edges=replay_max_edges,
            )
            if verification.status is CertificateVerificationStatus.LIMIT_REACHED:
                return ClockworkRetractionSearchResult(
                    ClockworkSearchStatus.LIMIT_REACHED,
                    None,
                    checked,
                    attempts,
                    verification.reason,
                    states_explored,
                    assignments_tried,
                )
            if verification.status is CertificateVerificationStatus.INVALID:
                raise ValueError(
                    "Provided iterate provenance failed independent verification: "
                    f"{verification.reason}"
                )
            return ClockworkRetractionSearchResult(
                ClockworkSearchStatus.FOUND,
                certificate,
                checked,
                attempts,
                "Verified retraction onto an induced divergent clockwork.",
                states_explored,
                assignments_tried,
            )

    status = ClockworkSearchStatus.EXHAUSTED
    reason = "No divergent clockwork retract was found within the search bounds."
    if checked < candidates_total or recognition_incomplete or retraction_incomplete:
        status = ClockworkSearchStatus.LIMIT_REACHED
        reason = (
            "Search was incomplete due to a candidate, recognition, "
            "or retraction limit."
        )
    return ClockworkRetractionSearchResult(
        status,
        None,
        checked,
        attempts,
        reason,
        states_explored,
        assignments_tried,
    )
