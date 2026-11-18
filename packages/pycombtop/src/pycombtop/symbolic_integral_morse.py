"""Integral homology from a bounded ZDD implementation of Dong's matching.

A verified sequential matching gives an acyclic Morse model. Its integral
cellular differential is computed via signed gradient paths, not by
enumerating all original simplices. A wedge is claimed only when the
cellular attaching maps are determined by degree (or only one dimension
of positive critical cells remains).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import networkx as nx

from ._symbolic_morse_gradient import _MorseLimit, morse_boundaries
from .simplex import _stable_vertex_key
from .symbolic_dong import _DiagramLimit, _ZDD


@dataclass(frozen=True)
class SymbolicIntegralMorseCertificate:
    """Exact reduced integral homology and optionally a wedge certificate.

    ``boundaries[d]`` holds rows of the Morse boundary from C_d to C_(d-1),
    in increasing critical bitmask order; absent matrices are zero.
    ``torsion[d]`` gives invariant factors of H_d beyond its free rank.
    """

    critical: dict[int, int]
    boundaries: dict[int, tuple[tuple[int, ...], ...]]
    betti: dict[int, int]
    torsion: dict[int, tuple[int, ...]]
    wedge_spheres: dict[int, int] | None
    total_faces: int
    diagram_nodes: int
    gradient_states: int
    critical_masks: tuple[int, ...]


def symbolic_integral_morse_certificate(
    graph: nx.Graph,
    *,
    vertex_order: Iterable[Hashable] | None = None,
    max_vertices: int = 80,
    max_nodes: int = 250_000,
    max_critical: int = 64,
    max_gradient_states: int = 200_000,
) -> SymbolicIntegralMorseCertificate | None:
    """Compute an exact integral Morse complex without listing its faces.

    A resource limit returns None. For more than two critical dimensions,
    homology is still exact but no wedge homotopy type is claimed. For two
    consecutive dimensions d,d+1, d>=2, Smith factors all equal to one
    certify cancellation of all nonzero-degree attaching maps.
    """
    if min(max_nodes, max_critical, max_gradient_states) < 1 or max_vertices < 0:
        raise ValueError("resource limits must be positive; max_vertices >= 0")
    n = len(graph)
    if n > max_vertices:
        return None
    if vertex_order is None:
        vertices = (
            sorted(graph) if all(isinstance(v, int) for v in graph)
            else sorted(graph, key=_stable_vertex_key)
        )
    else:
        vertices = list(vertex_order)
        if len(vertices) != n or len(set(vertices)) != n or set(vertices) != set(graph):
            raise ValueError("vertex_order must permute graph vertices")

    diagram = None
    try:
        diagram = _ZDD(n, max_nodes)
        root = diagram.clique_faces(graph, vertices)
        diagram.without_vertices.cache_clear()
        total = sum(diagram.face_size_counts(root))
        diagram.face_size_counts.cache_clear()
        roots = [root]
        for pivot in range(n):
            root = diagram.pivot(root, pivot)
            roots.append(root)
            diagram.difference.cache_clear()
            diagram.pivot.cache_clear()
            diagram.contains_empty.cache_clear()
            diagram.remove_empty.cache_clear()
        sizes = diagram.face_size_counts(root)
        diagram.face_size_counts.cache_clear()
        if sum(sizes) > max_critical:
            return None
        masks = tuple(sorted(diagram.masks(root)))
        if 0 in masks:
            return None  # augmented empty face must be matched
        faces: dict[int, list[int]] = defaultdict(list)
        for mask in masks:
            faces[mask.bit_count() - 1].append(mask)
        critical = dict(Counter(mask.bit_count() - 1 for mask in masks))
        boundaries, ranks, torsion, states = morse_boundaries(
            diagram, roots, faces, max_gradient_states
        )
        betti = {
            d: c - ranks.get(d, 0) - ranks.get(d + 1, 0)
            for d, c in critical.items()
        }
        if any(c < 0 for c in betti.values()):
            raise ArithmeticError("Morse boundary ranks are inconsistent")
        degrees = sorted(critical)
        wedge = None
        if len(degrees) <= 1:
            wedge = dict(critical)
        elif len(degrees) == 2 and degrees[0] >= 2 and degrees[1] == degrees[0] + 1:
            if not torsion:  # integer Smith invariants are all 1 or 0
                wedge = {d: b for d, b in betti.items() if b}
        return SymbolicIntegralMorseCertificate(
            critical=critical,
            boundaries=boundaries,
            betti={d: b for d, b in betti.items() if b},
            torsion=torsion,
            wedge_spheres=wedge,
            total_faces=total,
            diagram_nodes=len(diagram.variables) - 2,
            gradient_states=states,
            critical_masks=masks,
        )
    except (MemoryError, RecursionError, _DiagramLimit, _MorseLimit):
        return None
    finally:
        if diagram is not None:
            diagram.without_vertices.cache_clear()
            diagram.contains_empty.cache_clear()
            diagram.remove_empty.cache_clear()
            diagram.difference.cache_clear()
            diagram.pivot.cache_clear()
            diagram.face_size_counts.cache_clear()
