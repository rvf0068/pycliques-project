"""Replayable ordinary graph folds proving an attaching map is nullhomotopic.

If D is a vertex deletion and L its link graph, a sequence of ordinary
dominated-vertex folds D -> D' induces a homotopy equivalence on flag
complexes. If the image of L in D' lies in a cone, then L -> D is
nullhomotopic -- without a dimension restriction on the source spheres.
Every vertex fold, its image map and the cone apex are replayed exactly.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import networkx as nx

from .contractible_carriers import (
    CarrierMove,
    bounded_contractible_carrier,
    verify_contractible_carrier,
)
from .wedge_certificates import certified_flag_wedge


@dataclass(frozen=True)
class ConeImageCertificate:
    """Ordinary fold sequence, certified target type, and cone-image data."""

    moves: tuple[tuple[Hashable, Hashable], ...]
    spheres: dict[int, int]
    image_vertices: tuple[Hashable, ...]
    apex: Hashable


@dataclass(frozen=True)
class ContractibleCarrierCertificate:
    """A folded target plus a verified contractible carrier of link images.

    The carrier is the induced graph on carrier_vertices, containing all
    image_vertices. Its s-dismantling moves leave exactly survivor.
    This strictly generalizes the cone-image criterion.
    """

    moves: tuple[tuple[Hashable, Hashable], ...]
    spheres: dict[int, int]
    image_vertices: tuple[Hashable, ...]
    carrier_vertices: tuple[Hashable, ...]
    carrier_moves: tuple[CarrierMove, ...]
    survivor: Hashable


def _replay_fold_map(
    deletion: nx.Graph,
    link_vertices: Sequence[Hashable],
    moves: Sequence[tuple[Hashable, Hashable]],
) -> tuple[nx.Graph, tuple[Hashable, ...]] | None:
    """Replay ordinary folds, tracking all original link vertex images."""
    current = deletion.copy()
    if not set(link_vertices) <= set(current):
        return None
    images = {v: v for v in link_vertices}
    try:
        for move in moves:
            if not isinstance(move, (tuple, list)) or len(move) != 2:
                return None
            removed, dominator = move
            if removed not in current or dominator not in current:
                return None
            if not current.has_edge(removed, dominator):
                return None
            closed_removed = set(current[removed]) | {removed}
            closed_dominator = set(current[dominator]) | {dominator}
            if not closed_removed <= closed_dominator:
                return None
            for source, target in images.items():
                if target == removed:
                    images[source] = dominator
            current.remove_node(removed)
    except (TypeError, ValueError):
        return None
    image = set(images.values())
    return current, tuple(v for v in current if v in image)


def verify_cone_image_certificate(
    deletion: nx.Graph,
    link_vertices: Sequence[Hashable],
    certificate: ConeImageCertificate,
) -> bool:
    """Independent check of all folds, wedge type, image, and cone apex."""
    try:
        replay = _replay_fold_map(deletion, link_vertices, certificate.moves)
        if replay is None:
            return False
        core, image = replay
        if image != certificate.image_vertices:
            return False
        if certified_flag_wedge(core) != certificate.spheres:
            return False
        if not image or certificate.apex not in image:
            return False
        return all(
            v == certificate.apex or core.has_edge(v, certificate.apex) for v in image
        )
    except (TypeError, ValueError):
        return False


def _bounded_folded_target(
    deletion: nx.Graph,
    link_vertices: Sequence[Hashable],
    *,
    max_vertices: int,
    max_checks: int,
    max_moves: int,
) -> (
    tuple[
        tuple[tuple[Hashable, Hashable], ...],
        nx.Graph,
        tuple[Hashable, ...],
        dict[int, int],
    ]
    | None
):
    """One bounded ordinary-fold search shared by all image certificates.

    The bounds apply to closed-neighborhood domination comparisons.
    An ordinary fold defines a *simplicial retraction* of flag
    complexes, unlike a general vertex s-dismantling.
    """
    try:
        if (
            deletion.number_of_nodes() > max_vertices
            or max_checks < 0
            or max_moves < 0
            or not deletion
            or not nx.is_connected(deletion)
            or not set(link_vertices) <= set(deletion)
        ):
            return None
        current = deletion.copy()
        moves: list[tuple[Hashable, Hashable]] = []
        checks = 0
        for _ in range(max_moves):
            found = False
            for vertex in tuple(current):
                removed_closed = set(current[vertex]) | {vertex}
                for neighbor in current.neighbors(vertex):
                    if checks >= max_checks:
                        return None
                    checks += 1
                    if removed_closed <= set(current[neighbor]) | {neighbor}:
                        moves.append((vertex, neighbor))
                        current.remove_node(vertex)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        spheres = certified_flag_wedge(current)
        if spheres is None or 0 in spheres:
            return None
        replay = _replay_fold_map(deletion, link_vertices, moves)
        if replay is None:
            return None
        core, image = replay
        if not image:
            return None
        return tuple(moves), core, image, spheres
    except (TypeError, ValueError, KeyError):
        return None


def _cone_certificate(
    deletion: nx.Graph,
    link_vertices: Sequence[Hashable],
    result: tuple[
        tuple[tuple[Hashable, Hashable], ...],
        nx.Graph,
        tuple[Hashable, ...],
        dict[int, int],
    ],
) -> ConeImageCertificate | None:
    moves, core, image, spheres = result
    for apex in image:
        if all(v == apex or core.has_edge(v, apex) for v in image):
            certificate = ConeImageCertificate(moves, spheres, image, apex)
            if verify_cone_image_certificate(deletion, link_vertices, certificate):
                return certificate
            return None
    return None


def bounded_cone_image_certificate(
    deletion: nx.Graph,
    link_vertices: Sequence[Hashable],
    *,
    max_vertices: int = 24,
    max_checks: int = 1024,
    max_moves: int = 18,
) -> ConeImageCertificate | None:
    """Backward-compatible cone-image certificate with a bounded fold run."""
    folded = _bounded_folded_target(
        deletion,
        link_vertices,
        max_vertices=max_vertices,
        max_checks=max_checks,
        max_moves=max_moves,
    )
    if folded is None:
        return None
    return _cone_certificate(deletion, link_vertices, folded)


def verify_contractible_image_certificate(
    deletion: nx.Graph,
    link_vertices: Sequence[Hashable],
    certificate: ContractibleCarrierCertificate,
) -> bool:
    """Independently replay the deletion retraction and carrier contraction."""
    try:
        replay = _replay_fold_map(deletion, link_vertices, certificate.moves)
        if replay is None:
            return False
        core, image = replay
        if image != certificate.image_vertices:
            return False
        if certified_flag_wedge(core) != certificate.spheres:
            return False
        selected = set(certificate.carrier_vertices)
        if (
            not selected <= set(core)
            or not set(image) <= selected
            or tuple(v for v in core if v in selected) != certificate.carrier_vertices
        ):
            return False
        carrier = core.subgraph(certificate.carrier_vertices).copy()
        return verify_contractible_carrier(
            carrier, certificate.carrier_moves, certificate.survivor
        )
    except (TypeError, ValueError, KeyError):
        return False


def bounded_null_image_certificate(
    deletion: nx.Graph,
    link_vertices: Sequence[Hashable],
    *,
    max_vertices: int = 24,
    max_checks: int = 1024,
    max_moves: int = 18,
    max_carrier_vertices: int = 12,
    max_extra_vertices: int = 2,
    max_carrier_candidates: int = 36,
    max_carrier_checks: int = 64,
) -> ConeImageCertificate | ContractibleCarrierCertificate | None:
    """Bounded proof that the link image lies in a contractible subcomplex.

    Prefer the original cone witness. Otherwise check the image-induced
    graph and at most two additional carrier vertices, with explicit
    caps on candidates and local s-dismantling checks. Ordinary deletion
    folds run *once*, shared by both proof methods.
    """
    folded = _bounded_folded_target(
        deletion,
        link_vertices,
        max_vertices=max_vertices,
        max_checks=max_checks,
        max_moves=max_moves,
    )
    if folded is None:
        return None
    cone = _cone_certificate(deletion, link_vertices, folded)
    if cone is not None:
        return cone
    moves, core, image, spheres = folded
    carrier = bounded_contractible_carrier(
        core,
        image,
        max_carrier_vertices=max_carrier_vertices,
        max_extra_vertices=max_extra_vertices,
        max_candidates=max_carrier_candidates,
        max_checks_per_candidate=max_carrier_checks,
        max_moves=max_carrier_vertices - 1,
    )
    if carrier is None:
        return None
    carrier_vertices, carrier_moves, survivor = carrier
    certificate = ContractibleCarrierCertificate(
        moves, spheres, image, carrier_vertices, carrier_moves, survivor
    )
    if verify_contractible_image_certificate(deletion, link_vertices, certificate):
        return certificate
    return None
