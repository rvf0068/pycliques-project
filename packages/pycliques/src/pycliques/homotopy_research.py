"""Bounded structure and incidence certificates for clique-graph nerves."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
from typing import Any

import networkx as nx

from .cliques import clique_graph
from .homotopy_invariance import minimal_violation, witnesses_and_extension


class EnumerationStatus(Enum):
    """Whether all clique candidates were processed."""

    COMPLETE = "COMPLETE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class MissingFaceCertificate:
    """Incidence data for one inclusion-minimal non-Helly clique of K(G)."""

    clique_ids: tuple[int, ...]
    witness_cores: tuple[tuple[int, tuple[int, ...]], ...]
    witness_vertices: tuple[int, ...]
    extension_vertex_ids: tuple[int, ...]
    link_vertex_ids: tuple[int, ...]
    link_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HomotopyStructureResult:
    """Graph6-labelled structures and certificates from a bounded search."""

    graph6: str
    vertex_labels: tuple[str, ...]
    clique_labels: tuple[tuple[int, ...], ...]
    smallest_violation_ids: tuple[int, ...] | None
    missing_faces: tuple[MissingFaceCertificate, ...]
    conflict_edges: tuple[tuple[int, int], ...]
    explored_cliques: int
    status: EnumerationStatus
    diagnostics: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result with enum values rendered as strings."""
        result = asdict(self)
        result["status"] = self.status.value
        return result

    def to_json(self) -> str:
        """Return a compact JSON representation of this result."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def _graph6(graph: nx.Graph, vertex_ids: dict[Any, int]) -> str:
    relabelled = nx.relabel_nodes(graph, vertex_ids, copy=True)
    encoded: bytes = nx.to_graph6_bytes(
        relabelled, nodes=list(range(graph.number_of_nodes())), header=False
    )
    return encoded.decode("ascii").strip()


def _intersection(cliques: list[Any], vertices: set[Any]) -> set[Any]:
    if not cliques:
        return set(vertices)
    common = set(cliques[0])
    for clique in cliques[1:]:
        common.intersection_update(clique)
    return common


def _is_complete(graph: nx.Graph, vertices: set[Any]) -> bool:
    return all(
        graph.has_edge(first, second) for first, second in combinations(vertices, 2)
    )


def homotopy_structure_certificates(
    graph: nx.Graph, *, max_cliques: int = 100_000
) -> HomotopyStructureResult:
    """Enumerate minimal missing nerve faces and their incidence structures.

    The budget counts cliques enumerated in K(G). An exhausted budget returns
    the certificates found so far with ``INCONCLUSIVE`` status.
    """
    if max_cliques < 1:
        raise ValueError("max_cliques must be positive")

    source_vertices = list(graph.nodes())
    source_ids = {vertex: index for index, vertex in enumerate(source_vertices)}
    kg = clique_graph(graph)
    if kg is None:
        raise RuntimeError("unbounded clique-graph construction unexpectedly failed")

    ordered_cliques = sorted(
        kg.nodes(),
        key=lambda clique: tuple(sorted(source_ids[vertex] for vertex in clique)),
    )
    clique_ids = {clique: index for index, clique in enumerate(ordered_cliques)}
    smallest_violation = minimal_violation(kg)
    smallest_violation_ids = (
        tuple(sorted(clique_ids[clique] for clique in smallest_violation))
        if smallest_violation is not None
        else None
    )
    clique_labels = tuple(
        tuple(sorted(source_ids[vertex] for vertex in clique))
        for clique in ordered_cliques
    )

    certificates: list[MissingFaceCertificate] = []
    explored = 0
    status = EnumerationStatus.COMPLETE
    for clique_tuple in nx.enumerate_all_cliques(kg):
        if explored >= max_cliques:
            status = EnumerationStatus.INCONCLUSIVE
            break
        explored += 1
        face = frozenset(clique_tuple)
        common = _intersection(list(face), set(source_vertices))
        if common:
            continue

        core_by_removed = {
            removed: _intersection(
                [clique for clique in face if clique != removed],
                set(source_vertices),
            )
            for removed in face
        }
        if not all(core_by_removed.values()):
            continue

        ordered_face = sorted(face, key=clique_ids.__getitem__)
        helper_witnesses, _ = witnesses_and_extension(graph, kg, ordered_face)
        if any(
            witness not in core_by_removed[clique]
            for witness, clique in zip(helper_witnesses, ordered_face, strict=True)
        ):
            raise RuntimeError("witness helper returned a vertex outside its core")
        witnesses = [
            min(core_by_removed[clique], key=source_ids.__getitem__)
            for clique in ordered_face
        ]
        extension = next(
            maximal_clique
            for maximal_clique in sorted(
                nx.find_cliques(graph),
                key=lambda clique: tuple(
                    sorted(source_ids[vertex] for vertex in clique)
                ),
            )
            if set(witnesses) <= set(maximal_clique)
        )
        witness_cores = tuple(
            (
                clique_ids[removed],
                tuple(
                    sorted(source_ids[vertex] for vertex in core_by_removed[removed])
                ),
            )
            for removed in ordered_face
        )
        link_vertices = {
            vertex
            for vertex in kg
            if vertex not in face
            and all(kg.has_edge(vertex, member) for member in face)
        }
        link_vertex_ids = tuple(sorted(clique_ids[vertex] for vertex in link_vertices))
        link_edges = tuple(
            sorted(
                (min(clique_ids[u], clique_ids[v]), max(clique_ids[u], clique_ids[v]))
                for u, v in kg.subgraph(link_vertices).edges()
            )
        )
        certificates.append(
            MissingFaceCertificate(
                clique_ids=tuple(clique_ids[clique] for clique in ordered_face),
                witness_cores=witness_cores,
                witness_vertices=tuple(source_ids[vertex] for vertex in witnesses),
                extension_vertex_ids=tuple(
                    sorted(source_ids[vertex] for vertex in extension)
                ),
                link_vertex_ids=link_vertex_ids,
                link_edges=link_edges,
            )
        )

    certificates.sort(
        key=lambda certificate: (
            len(certificate.clique_ids),
            certificate.clique_ids,
        )
    )
    conflict_edges = tuple(
        (first, second)
        for first, second in combinations(range(len(certificates)), 2)
        if _is_complete(
            kg,
            {
                ordered_cliques[index]
                for index in (
                    *certificates[first].clique_ids,
                    *certificates[second].clique_ids,
                )
            },
        )
    )
    diagnostics = (
        "All clique candidates were enumerated."
        if status is EnumerationStatus.COMPLETE
        else f"Stopped after max_cliques={max_cliques}; results may be incomplete."
    )
    return HomotopyStructureResult(
        graph6=_graph6(graph, source_ids),
        vertex_labels=tuple(
            f"{index}:{type(vertex).__module__}.{type(vertex).__qualname__}"
            for index, vertex in enumerate(source_vertices)
        ),
        clique_labels=clique_labels,
        smallest_violation_ids=smallest_violation_ids,
        missing_faces=tuple(certificates),
        conflict_edges=conflict_edges,
        explored_cliques=explored,
        status=status,
        diagnostics=diagnostics,
    )


def verify_homotopy_structure_certificates(
    graph: nx.Graph, result: HomotopyStructureResult
) -> bool:
    """Check all face, core, link, extension, and conflict incidences."""
    source_vertices = list(graph.nodes())
    source_ids = {vertex: index for index, vertex in enumerate(source_vertices)}
    if result.graph6 != _graph6(graph, source_ids):
        return False
    if result.vertex_labels != tuple(
        f"{index}:{type(vertex).__module__}.{type(vertex).__qualname__}"
        for index, vertex in enumerate(source_vertices)
    ):
        return False

    kg = clique_graph(graph)
    if kg is None:
        return False
    ordered_cliques = sorted(
        kg.nodes(),
        key=lambda clique: tuple(sorted(source_ids[vertex] for vertex in clique)),
    )
    clique_ids = {clique: index for index, clique in enumerate(ordered_cliques)}
    if result.clique_labels != tuple(
        tuple(sorted(source_ids[vertex] for vertex in clique))
        for clique in ordered_cliques
    ):
        return False
    clique_by_id = dict(enumerate(ordered_cliques))
    expected_conflicts: list[tuple[int, int]] = []
    seen_faces: set[tuple[int, ...]] = set()
    if result.smallest_violation_ids is not None and (
        not result.smallest_violation_ids
        or any(index not in clique_by_id for index in result.smallest_violation_ids)
    ):
        return False

    for certificate in result.missing_faces:
        ids = certificate.clique_ids
        if (
            not ids
            or len(ids) != len(set(ids))
            or any(index not in clique_by_id for index in ids)
            or ids in seen_faces
        ):
            return False
        seen_faces.add(ids)
        face = [clique_by_id[index] for index in ids]
        if not _is_complete(kg, set(face)) or _intersection(face, set(source_vertices)):
            return False
        expected_cores = tuple(
            (
                removed_id,
                tuple(
                    sorted(
                        source_ids[vertex]
                        for vertex in _intersection(
                            [
                                clique
                                for clique_id, clique in zip(ids, face, strict=True)
                                if clique_id != removed_id
                            ],
                            set(source_vertices),
                        )
                    )
                ),
            )
            for removed_id in ids
        )
        if certificate.witness_cores != expected_cores or any(
            not core for _, core in expected_cores
        ):
            return False
        if (
            len(certificate.witness_vertices) != len(ids)
            or any(
                index < 0 or index >= len(source_vertices)
                for index in certificate.witness_vertices
            )
            or len(set(certificate.witness_vertices))
            != len(certificate.witness_vertices)
        ):
            return False
        if any(
            witness not in core
            for witness, (_, core) in zip(
                certificate.witness_vertices, expected_cores, strict=True
            )
        ):
            return False

        if (
            not certificate.extension_vertex_ids
            or len(set(certificate.extension_vertex_ids))
            != len(certificate.extension_vertex_ids)
            or any(
                index < 0 or index >= len(source_vertices)
                for index in certificate.extension_vertex_ids
            )
        ):
            return False
        extension_vertices = {
            source_vertices[index] for index in certificate.extension_vertex_ids
        }
        if not _is_complete(graph, extension_vertices):
            return False
        if not any(
            set(maximal_clique) == extension_vertices
            for maximal_clique in nx.find_cliques(graph)
        ):
            return False
        if (
            not set(source_vertices[index] for index in certificate.witness_vertices)
            <= extension_vertices
        ):
            return False

        link_vertices = {
            vertex
            for vertex in kg
            if vertex not in face
            and all(kg.has_edge(vertex, member) for member in face)
        }
        expected_link_vertices = tuple(
            sorted(clique_ids[vertex] for vertex in link_vertices)
        )
        if certificate.link_vertex_ids != expected_link_vertices:
            return False
        expected_link_edges = tuple(
            sorted(
                (
                    min(clique_ids[u], clique_ids[v]),
                    max(clique_ids[u], clique_ids[v]),
                )
                for u, v in kg.subgraph(link_vertices).edges()
            )
        )
        if certificate.link_edges != expected_link_edges:
            return False

    if result.missing_faces != tuple(
        sorted(
            result.missing_faces,
            key=lambda certificate: (
                len(certificate.clique_ids),
                certificate.clique_ids,
            ),
        )
    ):
        return False

    for first, second in combinations(range(len(result.missing_faces)), 2):
        union = {
            clique_by_id[index]
            for index in (
                *result.missing_faces[first].clique_ids,
                *result.missing_faces[second].clique_ids,
            )
        }
        if _is_complete(kg, union):
            expected_conflicts.append((first, second))
    if result.conflict_edges != tuple(expected_conflicts):
        return False

    if result.status is EnumerationStatus.COMPLETE:
        expected_faces: set[tuple[int, ...]] = set()
        for clique in nx.enumerate_all_cliques(kg):
            candidate_face = frozenset(clique)
            if _intersection(list(candidate_face), set(source_vertices)):
                continue
            if all(
                _intersection(
                    [member for member in candidate_face if member != removed],
                    set(source_vertices),
                )
                for removed in candidate_face
            ):
                expected_faces.add(
                    tuple(sorted(clique_ids[member] for member in candidate_face))
                )
        returned_faces = {
            certificate.clique_ids for certificate in result.missing_faces
        }
        if expected_faces != returned_faces:
            return False
        if result.smallest_violation_ids is not None and (
            result.smallest_violation_ids not in returned_faces
        ):
            return False
        return True
    return True
