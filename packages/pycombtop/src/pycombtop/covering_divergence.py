"""Replay certificates transferring clique divergence through triangular covers."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
from pycliques.cliques import Clique, clique_graph
from pycliques.clockwork_search import (
    CertificateVerificationResult,
    CertificateVerificationStatus,
)
from pycliques.small import Certificate, verify_theorem_2_6_certificate

from .fundamental_group import covering_graph, fundamental_group


@dataclass(frozen=True)
class TriangularCoverDivergenceCertificate:
    """Provenance for a clockwork proof transferred across a finite cover.

    The graphs and maps are explicit Python data, not executable serialized
    content. The verifier checks the cover projection, local triangular
    condition, deck action, reconstructed clique graph, and inner theorem
    certificate before returning ``VERIFIED``.
    """

    base_graph: nx.Graph
    cover_graph: nx.Graph
    projection: Mapping[Hashable, Hashable]
    deck_transformation: Mapping[Hashable, Hashable]
    theorem_certificate: Certificate
    subgroup_words: tuple[tuple[int, ...], ...] = ()
    voltage_cocycle: tuple[tuple[Hashable, Hashable, int], ...] = ()
    target_level: str = "clique_graph"
    clique_bound: int | float | None = 100
    clique_max_edges: int | None = 2000
    deck_radius: int = 4


def _simple_undirected(graph: object) -> bool:
    return (
        isinstance(graph, nx.Graph)
        and not graph.is_directed()
        and not graph.is_multigraph()
        and nx.number_of_selfloops(graph) == 0
    )


def _same_labeled_graph(first: nx.Graph, second: nx.Graph) -> bool:
    return set(first) == set(second) and {
        frozenset(edge) for edge in first.edges()
    } == {frozenset(edge) for edge in second.edges()}


def _verify_triangular_projection(
    base: nx.Graph,
    cover: nx.Graph,
    projection: Mapping[Hashable, Hashable],
) -> str | None:
    if set(projection) != set(cover):
        return "Projection is not defined on every cover vertex."
    if not set(projection.values()) <= set(base):
        return "Projection maps a cover vertex outside the base graph."
    fibers: dict[Hashable, list[Hashable]] = {vertex: [] for vertex in base}
    for cover_vertex, base_vertex in projection.items():
        fibers[base_vertex].append(cover_vertex)
    if any(len(fiber) != 2 for fiber in fibers.values()):
        return "Projection is not a surjective two-sheeted map."
    if not nx.is_connected(cover):
        return "The cover graph is disconnected."
    for left, right in cover.edges():
        if not base.has_edge(projection[left], projection[right]):
            return "A cover edge does not project to a base edge."

    for vertex in cover:
        image = projection[vertex]
        closed_cover = {vertex, *cover.neighbors(vertex)}
        closed_base = {image, *base.neighbors(image)}
        projected = [projection[neighbor] for neighbor in closed_cover]
        if len(set(projected)) != len(closed_cover) or set(projected) != closed_base:
            return "Projection is not bijective on a closed neighborhood."
        for left, right in combinations(closed_cover, 2):
            if cover.has_edge(left, right) != base.has_edge(
                projection[left], projection[right]
            ):
                return "Projection does not preserve the induced closed neighborhood."
    return None


def verify_triangular_cover_divergence_certificate(
    certificate: TriangularCoverDivergenceCertificate,
) -> CertificateVerificationResult:
    """Independently verify every covering and Theorem 2.6 proof step."""
    invalid = CertificateVerificationStatus.INVALID
    if not _simple_undirected(certificate.base_graph):
        return CertificateVerificationResult(invalid, "Base graph must be simple.")
    if not _simple_undirected(certificate.cover_graph):
        return CertificateVerificationResult(invalid, "Cover graph must be simple.")
    if certificate.base_graph.number_of_nodes() == 0:
        return CertificateVerificationResult(invalid, "Base graph must be nonempty.")
    if not nx.is_connected(certificate.base_graph):
        return CertificateVerificationResult(invalid, "Base graph must be connected.")
    if certificate.target_level != "clique_graph":
        return CertificateVerificationResult(
            invalid, "Only a clique-graph Theorem 2.6 target is supported."
        )
    if certificate.deck_radius < 1:
        return CertificateVerificationResult(invalid, "Deck radius must be positive.")
    if certificate.clique_bound is not None and certificate.clique_bound < 0:
        return CertificateVerificationResult(invalid, "Clique bound is negative.")
    if certificate.clique_max_edges is not None and certificate.clique_max_edges < 0:
        return CertificateVerificationResult(invalid, "Clique edge bound is negative.")

    projection_error = _verify_triangular_projection(
        certificate.base_graph, certificate.cover_graph, certificate.projection
    )
    if projection_error is not None:
        return CertificateVerificationResult(invalid, projection_error)
    if certificate.subgroup_words:
        expected_labels = set(range(certificate.base_graph.number_of_nodes()))
        if set(certificate.base_graph) != expected_labels:
            return CertificateVerificationResult(
                invalid,
                "Subgroup replay requires canonical integer base labels 0..n-1.",
            )
        record = fundamental_group(certificate.base_graph)
        if any(
            not isinstance(generator, int)
            or generator == 0
            or abs(generator) > record.rank
            for word in certificate.subgroup_words
            for generator in word
        ):
            return CertificateVerificationResult(
                invalid, "Subgroup words contain an invalid generator index."
            )
        try:
            subgroup_cover = covering_graph(
                certificate.base_graph,
                [list(word) for word in certificate.subgroup_words],
                rec=record,
                max_cosets=2,
            )
        except RuntimeError as error:
            return CertificateVerificationResult(
                invalid, f"Subgroup does not define the recorded double cover: {error}"
            )
        if not _same_labeled_graph(subgroup_cover, certificate.cover_graph):
            return CertificateVerificationResult(
                invalid, "Subgroup generators do not reconstruct the supplied cover."
            )
        expected_projection = {vertex: vertex[0] for vertex in subgroup_cover}
        if dict(certificate.projection) != expected_projection:
            return CertificateVerificationResult(
                invalid, "Subgroup cover projection differs from its canonical map."
            )
    if certificate.voltage_cocycle:
        voltages: dict[frozenset[Hashable], int] = {}
        for first, second, value in certificate.voltage_cocycle:
            edge = frozenset((first, second))
            if (
                len(edge) != 2
                or not certificate.base_graph.has_edge(first, second)
                or value not in (0, 1)
                or edge in voltages
            ):
                return CertificateVerificationResult(
                    invalid, "Voltage cocycle has an invalid or duplicate edge entry."
                )
            voltages[edge] = value
        base_edges = {frozenset(edge) for edge in certificate.base_graph.edges()}
        if set(voltages) != base_edges:
            return CertificateVerificationResult(
                invalid, "Voltage cocycle does not cover every base edge."
            )
        for first, second in certificate.base_graph.edges():
            change = voltages[frozenset((first, second))]
            for sheet in (0, 1):
                if not certificate.cover_graph.has_edge(
                    (first, sheet), (second, sheet ^ change)
                ):
                    return CertificateVerificationResult(
                        invalid,
                        "Voltage cocycle does not reconstruct the supplied cover.",
                    )

    cover = certificate.cover_graph
    projection = certificate.projection
    deck = certificate.deck_transformation
    if set(deck) != set(cover) or set(deck.values()) != set(cover):
        return CertificateVerificationResult(
            invalid, "Deck transformation is not a permutation of cover vertices."
        )
    if any(deck.get(deck[vertex]) != vertex for vertex in cover):
        return CertificateVerificationResult(invalid, "Deck map is not an involution.")
    if any(projection[deck[vertex]] != projection[vertex] for vertex in cover):
        return CertificateVerificationResult(
            invalid, "Deck map does not preserve the covering projection."
        )
    if any(
        not cover.has_edge(deck[left], deck[right]) for left, right in cover.edges()
    ):
        return CertificateVerificationResult(
            invalid, "Deck map is not a cover-graph automorphism."
        )

    cover_distances = dict(nx.all_pairs_shortest_path_length(cover))
    if any(
        cover_distances[vertex].get(deck[vertex], math.inf) < certificate.deck_radius
        for vertex in cover
    ):
        return CertificateVerificationResult(
            invalid, "Deck map fails its recorded coaffinity radius."
        )

    target = clique_graph(
        cover,
        math.inf if certificate.clique_bound is None else certificate.clique_bound,
        max_edges=certificate.clique_max_edges,
    )
    if target is None:
        return CertificateVerificationResult(
            CertificateVerificationStatus.LIMIT_REACHED,
            "Clique-graph reconstruction exceeded its configured vertex or edge bound.",
        )
    inner = certificate.theorem_certificate
    if not isinstance(inner.radius, int) or certificate.deck_radius < inner.radius:
        return CertificateVerificationResult(
            invalid,
            "Cover deck radius is smaller than the theorem's required radius.",
        )
    if not isinstance(inner.target, nx.Graph) or not _same_labeled_graph(
        inner.target, target
    ):
        return CertificateVerificationResult(
            invalid, "Theorem certificate target is not the reconstructed clique graph."
        )
    expected_target_action = {
        clique: Clique(deck[vertex] for vertex in clique) for clique in target
    }
    if inner.target_coaffination != expected_target_action:
        return CertificateVerificationResult(
            invalid, "Theorem target action is not induced by the deck map."
        )

    target_distances = dict(nx.all_pairs_shortest_path_length(target))
    if any(
        target_distances[clique].get(expected_target_action[clique], math.inf)
        < certificate.deck_radius
        for clique in target
    ):
        return CertificateVerificationResult(
            invalid, "Induced clique-graph action fails its coaffinity radius."
        )

    theorem_result = verify_theorem_2_6_certificate(target, inner)
    if theorem_result.status is not CertificateVerificationStatus.VERIFIED:
        return theorem_result
    return CertificateVerificationResult(
        CertificateVerificationStatus.VERIFIED,
        "Triangular cover and induced Theorem 2.6 divergence certificate verified.",
    )
