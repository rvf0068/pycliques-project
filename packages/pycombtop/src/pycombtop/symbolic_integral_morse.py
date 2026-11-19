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


def _wedge_after_one_circle_cancellation(
    critical: dict[int, int],
    boundaries: dict[int, tuple[tuple[int, ...], ...]],
    betti: dict[int, int],
    torsion: dict[int, tuple[int, ...]],
) -> dict[int, int] | None:
    """Certify a 2/3-dimensional wedge when the Morse 1-skeleton has one loop.

    The caller has an acyclic *augmented* Morse matching with the empty
    simplex paired. Restoring its partner adds the base 0-cell: the
    unaugmented Morse 1-skeleton has c0+1 vertices and c1 edges.
    Requiring c1=c0+1 and a full-unit-rank d1 proves this graph is
    connected with precisely one independent loop. Requiring d2 to
    have one unit Smith factor makes the 2-cell attaching degrees on
    that loop relatively prime, so the resulting fundamental group
    Z / gcd(degrees) is trivial.

    Its simply connected 2-skeleton is a wedge of 2-spheres by
    Hurewicz and Whitehead. The exact d3 then represents the 3-cell
    attaching maps in pi2=H2. Unit Smith factors permit cellular
    cancellation, leaving wedge summands in dimensions 2 and 3.
    No conclusion is drawn for multiple independent loops.
    """
    if torsion or any(d not in {0, 1, 2, 3} or c <= 0 for d, c in critical.items()):
        return None
    c0 = critical.get(0, 0)
    c1 = critical.get(1, 0)
    c2 = critical.get(2, 0)
    c3 = critical.get(3, 0)
    if c1 != c0 + 1 or c2 == 0:
        return None
    needed = {2}
    if c0:
        needed.add(1)
    if c3:
        needed.add(3)
    if set(boundaries) != needed:
        return None

    def unit_rank(degree: int, rows: int, cols: int) -> int | None:
        matrix = boundaries.get(degree)
        if (
            matrix is None
            or len(matrix) != rows
            or any(len(row) != cols for row in matrix)
        ):
            return None
        normal = smith_normal_form(Matrix(matrix), domain=ZZ)
        factors = [
            abs(int(normal[i, i]))
            for i in range(min(rows, cols))
            if normal[i, i] != 0
        ]
        return len(factors) if all(value == 1 for value in factors) else None

    if c0 and unit_rank(1, c0, c1) != c0:
        return None
    if unit_rank(2, c1, c2) != 1:
        return None
    r3 = 0
    if c3:
        rank = unit_rank(3, c2, c3)
        if rank is None:
            return None
        r3 = rank

    # Reject fabricated boundary data even when its individual Smith
    # factors are all units. The production Morse pipeline also checks d²=0.
    for low, high in ((1, 2), (2, 3)):
        if low in boundaries and high in boundaries:
            if any(
                entry != 0
                for entry in Matrix(boundaries[low]) * Matrix(boundaries[high])
            ):
                return None

    h2 = c2 - 1 - r3
    h3 = c3 - r3
    if min(h2, h3) < 0:
        return None
    spheres = {d: n for d, n in ((2, h2), (3, h3)) if n}
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
    and the upper Smith factors are units. A separate one-circle
    criterion handles critical cells in dimensions 0 through 3 when
    the 1-skeleton has one loop and primitive 2- and 3-cell attaching
    degrees. Unsupported patterns give exact homology but no wedge.
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
            positive_betti = {d: b for d, b in betti.items() if b}
            wedge = _wedge_after_01_cancellation(
                critical, boundaries, positive_betti, torsion
            )
            if wedge is None:
                wedge = _wedge_after_one_circle_cancellation(
                    critical, boundaries, positive_betti, torsion
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
