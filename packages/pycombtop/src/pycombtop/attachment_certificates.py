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

from .wedge_certificates import certified_flag_wedge


@dataclass(frozen=True)
class ConeImageCertificate:
    """Ordinary fold sequence, certified target type, and cone-image data."""

    moves: tuple[tuple[Hashable, Hashable], ...]
    spheres: dict[int, int]
    image_vertices: tuple[Hashable, ...]
    apex: Hashable


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


def bounded_cone_image_certificate(
    deletion: nx.Graph,
    link_vertices: Sequence[Hashable],
    *,
    max_vertices: int = 24,
    max_checks: int = 1024,
    max_moves: int = 18,
) -> ConeImageCertificate | None:
    """Try at most max_checks domination comparisons and max_moves folds.

    Search only ordinary closed-neighborhood dominated vertices. Unlike
    arbitrary graph s-dismantlings, every such fold has an explicit
    simplicial retraction, needed to certify the link image as a cone.
    """
    if (
        deletion.number_of_nodes() > max_vertices
        or max_checks < 0
        or max_moves < 0
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
    for apex in image:
        if all(v == apex or core.has_edge(v, apex) for v in image):
            result = ConeImageCertificate(tuple(moves), spheres, image, apex)
            if verify_cone_image_certificate(deletion, link_vertices, result):
                return result
            return None
    return None
