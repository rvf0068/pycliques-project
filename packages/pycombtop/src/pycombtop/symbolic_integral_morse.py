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
from sympy import ZZ, Matrix
from sympy.matrices.normalforms import smith_normal_form

from ._symbolic_morse_gradient import _MorseLimit, morse_boundaries
from .simplex import _stable_vertex_key
from .symbolic_dong import _ZDD, _DiagramLimit


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


def _wedge_after_01_cancellation(
    critical: dict[int, int],
    boundaries: dict[int, tuple[tuple[int, ...], ...]],
    betti: dict[int, int],
    torsion: dict[int, tuple[int, ...]],
) -> dict[int, int] | None:
    """Certify a wedge after cancelling a critical 0--1 Morse cell pair.

    Assumes the caller has already certified an *acyclic augmented*
    matching, paired the empty simplex, and computed exact integral
    cellular Morse boundaries. Restoring the empty-face partner yields
    two 0-cells joined by one 1-cell. A unit d_1 makes this a tree,
    which is collapsed before considering the positive-dimensional
    cells.

    The remaining critical cells must occur in one dimension d >= 2,
    or exactly two consecutive dimensions d,d+1. For d,d+1, all
    nonzero Smith invariants of d_(d+1) must be units. Homotopy
    classification then follows from pi_d(vee S^d) = H_d(vee S^d)
    for d >= 2. This is not a conclusion from homology alone.
    """
    if (
        torsion
        or critical.get(0) != 1
        or critical.get(1) != 1
        or any(d < 0 or count <= 0 for d, count in critical.items())
        or boundaries.get(1) not in (((1,),), ((-1,),))
    ):
        return None
    positive = sorted(d for d in critical if d >= 2)
    if (
        not positive
        or len(positive) > 2
        or (len(positive) == 2 and positive[1] != positive[0] + 1)
    ):
        return None

    base = positive[0]
    allowed = {1}
    if base == 2:
        allowed.add(2)
        if boundaries.get(2) != ((0,) * critical[2],):
            return None

    rank = 0
    if len(positive) == 2:
        top = positive[1]
        allowed.add(top)
        lower_count, upper_count = critical[base], critical[top]
        matrix = boundaries.get(top)
        if (
            matrix is None
            or len(matrix) != lower_count
            or any(len(row) != upper_count for row in matrix)
        ):
            return None
        smith = smith_normal_form(Matrix(matrix), domain=ZZ)
        diagonal = [
            abs(int(smith[i, i]))
            for i in range(min(lower_count, upper_count))
            if smith[i, i] != 0
        ]
        if any(value != 1 for value in diagonal):
            return None
        rank = len(diagonal)
    if set(boundaries) != allowed:
        return None

    expected = {base: critical[base] - rank}
    if len(positive) == 2:
        expected[positive[1]] = critical[positive[1]] - rank
    if any(count < 0 for count in expected.values()):
        return None
    spheres = {d: count for d, count in expected.items() if count}
    return spheres if spheres == betti else None


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

    A resource limit returns None. Two consecutive critical dimensions
    d,d+1, with d>=2 and unit Smith factors, determine a wedge. Four
    critical dimensions 0,1,d,d+1 may also give a certified wedge when
    the 0--1 boundary is a unit, all intermediate boundaries vanish,
    and the upper Smith factors are units. Unsupported patterns give
    exact homology but no asserted wedge.
    """
    if min(max_nodes, max_critical, max_gradient_states) < 1 or max_vertices < 0:
        raise ValueError("resource limits must be positive; max_vertices >= 0")
    n = len(graph)
    if n > max_vertices:
        return None
    if vertex_order is None:
        vertices = (
            sorted(graph)
            if all(isinstance(v, int) for v in graph)
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
        # An independent chain-complex consistency check: d^2 must vanish.
        for degree in boundaries:
            if degree - 1 in boundaries:
                product = Matrix(boundaries[degree - 1]) * Matrix(boundaries[degree])
                if any(entry != 0 for entry in product):
                    raise ArithmeticError("symbolic Morse differential has d^2 != 0")
        betti = {
            d: c - ranks.get(d, 0) - ranks.get(d + 1, 0) for d, c in critical.items()
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
        else:
            wedge = _wedge_after_01_cancellation(
                critical,
                boundaries,
                {d: b for d, b in betti.items() if b},
                torsion,
            )
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
