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

from ._symbolic_morse_gradient import _MorseLimit, matching_pivot, morse_boundaries
from ._symbolic_spherical import (
    AntipodalPairs,
    SphericalGenerationCertificate,
    spherical_generation,
)
from ._symbolic_steenrod import SymbolicSq2Certificate, compute_sq2
from .fundamental_group import connected_link_order
from .homotopy_expressions import (
    HomotopyType,
    SphereType,
    SuspendedHopfType,
    WedgeType,
)
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
    connectivity_order: tuple[Hashable, ...] | None = None
    vertex_order: tuple[Hashable, ...] = ()
    sq2: SymbolicSq2Certificate | None = None
    sq2_limited: bool = False
    sq2_homotopy_type: WedgeType | None = None
    spherical: SphericalGenerationCertificate | None = None

    @property
    def homotopy_type(self) -> WedgeType | None:
        """A certified structured type, never a type inferred from Betti alone."""
        if self.sq2_homotopy_type is not None:
            return self.sq2_homotopy_type
        if self.wedge_spheres is None:
            return None
        return WedgeType(
            tuple(
                (SphereType(d), b) for d, b in sorted(self.wedge_spheres.items()) if b
            )
        )


def _sq2_type(betti: dict[int, int], square: SymbolicSq2Certificate) -> WedgeType:
    """Assemble the type in the simply connected torsion-free stable range."""
    d, r = square.degree, square.rank
    counts = dict(betti)
    counts[d] -= r
    counts[d + 2] -= r
    terms: list[tuple[HomotopyType, int]] = [
        (SphereType(k), n) for k, n in sorted(counts.items()) if n
    ]
    if r:
        terms.append((SuspendedHopfType(d), r))
    return WedgeType(tuple(terms))


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
            abs(int(normal[i, i])) for i in range(min(rows, cols)) if normal[i, i] != 0
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
    max_sq2_states: int = 0,
    steenrod_degree: int | None = None,
    max_spherical_states: int = 0,
    spherical_seed: Iterable[AntipodalPairs] = (),
    spherical_search: bool = True,
) -> SymbolicIntegralMorseCertificate | None:
    """Compute an exact integral Morse complex without listing its faces.

    A resource limit returns None. Two consecutive critical dimensions
    d,d+1, with d>=2 and unit Smith factors, determine a wedge. Four
    critical dimensions 0,1,d,d+1 may also give a certified wedge when
    the 0--1 boundary is a unit, all intermediate boundaries vanish,
    and the upper Smith factors are units. A separate one-circle
    criterion handles critical cells in dimensions 0 through 3 when
    the 1-skeleton has one loop and primitive 2- and 3-cell attaching
    degrees. With one critical 1-cell, no critical 0-cells in the
    augmented matching, and only critical 2-cells otherwise, a zero
    d2 preserves the circle and gives a wedge of 1- and 2-spheres.
    For other critical-cell patterns, torsion-free integral homology in
    at most two consecutive degrees >=2 (or acyclicity), together with a
    verified connected-link order, also proves a wedge by relative
    Hurewicz and Whitehead. Unsupported patterns give exact homology
    but no wedge.

    Set max_sq2_states positive to attempt symbolic Sq^2 on torsion-free
    homology spanning degrees d through d+2, d>=3, with a connected-link
    order. The rank determines the number of suspended Hopf summands.
    An explicit steenrod_degree >=2 also computes the operation outside
    this range, without claiming a type there. A square limit retains
    the integral certificate and sets sq2_limited, never a zero matrix.

    A positive max_spherical_states enables integral spherical generation
    after the existing wedge and square criteria. Supply antipodal pairs
    from previous partial certificates as spherical_seed; set spherical_search
    false to project only those supplied spheres. All required degrees must
    generate the full integral lattice before a wedge is asserted.
    """
    if min(max_nodes, max_critical, max_gradient_states) < 1 or max_vertices < 0:
        raise ValueError("resource limits must be positive; max_vertices >= 0")
    if type(max_sq2_states) is not int or max_sq2_states < 0:
        raise ValueError("max_sq2_states must be a nonnegative integer")
    if steenrod_degree is not None and (
        type(steenrod_degree) is not int or steenrod_degree < 2 or not max_sq2_states
    ):
        raise ValueError(
            "steenrod_degree requires degree >=2 and a positive square budget"
        )
    if type(max_spherical_states) is not int or max_spherical_states < 0:
        raise ValueError("max_spherical_states must be a nonnegative integer")
    if type(spherical_search) is not bool:
        raise ValueError("spherical_search must be boolean")
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
        pivot_lookup = matching_pivot(diagram, roots)
        boundaries, ranks, torsion, states = morse_boundaries(
            diagram, roots, faces, max_gradient_states, pivot_lookup=pivot_lookup
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
        elif degrees == [1, 2] and critical[1] == 1 and ranks.get(2, 0) == 0:
            # The unaugmented CW model has one vertex and one loop.
            # Each 2-cell attaching map is classified by its degree in Z.
            wedge = dict(critical)
        else:
            positive_betti = {d: b for d, b in betti.items() if b}
            wedge = _wedge_after_01_cancellation(
                critical, boundaries, positive_betti, torsion
            )
            if wedge is None:
                wedge = _wedge_after_one_circle_cancellation(
                    critical, boundaries, positive_betti, torsion
                )
        connectivity_order = None
        if wedge is None and not torsion:
            support = sorted(d for d, rank in betti.items() if rank)
            if not support or (support[0] >= 2 and support[-1] - support[0] <= 1):
                connectivity_order = connected_link_order(graph)
                if connectivity_order is not None:
                    wedge = {d: b for d, b in betti.items() if b}
        square = None
        square_limited = False
        square_type = None
        support = sorted(d for d, rank in betti.items() if rank)
        in_range = (
            not torsion
            and bool(support)
            and support[0] >= 3
            and support[-1] - support[0] == 2
        )
        degree = steenrod_degree
        if max_sq2_states and in_range and (degree is None or degree == support[0]):
            connectivity_order = connected_link_order(graph)
            if connectivity_order is not None:
                degree = support[0]
        if degree is not None and max_sq2_states:
            try:
                square = compute_sq2(
                    faces, boundaries, pivot_lookup, degree, max_sq2_states
                )
            except (_MorseLimit, MemoryError, RecursionError):
                square_limited = True
            if (
                square is not None
                and in_range
                and degree == support[0]
                and connectivity_order is not None
            ):
                if (square.source_rank, square.target_rank) != (
                    betti[degree],
                    betti[degree + 2],
                ):
                    raise ArithmeticError("mod-2 and integral homology disagree")
                square_type = _sq2_type(betti, square)
                if square.rank == 0:
                    wedge = {d: b for d, b in betti.items() if b}
                else:
                    # Never publish a non-spherical type as a sphere wedge.
                    wedge = None
        spherical = None
        if (
            max_spherical_states
            and wedge is None
            and square_type is None
            and not torsion
            and support
            and support[0] >= 2
        ):
            connectivity_order = connected_link_order(graph)
            if connectivity_order is not None:
                spherical = spherical_generation(
                    graph,
                    vertices,
                    faces,
                    boundaries,
                    {d: b for d, b in betti.items() if b},
                    pivot_lookup,
                    max_spherical_states,
                    seed=spherical_seed,
                    search=spherical_search,
                )
                if spherical.complete:
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
            connectivity_order=connectivity_order,
            vertex_order=tuple(vertices),
            sq2=square,
            sq2_limited=square_limited,
            sq2_homotopy_type=square_type,
            spherical=spherical,
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


def verify_symbolic_sq2(
    graph: nx.Graph,
    certificate: SymbolicIntegralMorseCertificate,
    *,
    max_vertices: int = 80,
    max_nodes: int = 250_000,
    max_critical: int = 128,
    max_gradient_states: int = 200_000,
    max_sq2_states: int = 2_000_000,
) -> bool:
    """Replay a square certificate in its recorded order with bounded resources."""
    try:
        if certificate.sq2 is None:
            return False
        replay = symbolic_integral_morse_certificate(
            graph,
            vertex_order=certificate.vertex_order,
            max_vertices=max_vertices,
            max_nodes=max_nodes,
            max_critical=max_critical,
            max_gradient_states=max_gradient_states,
            max_sq2_states=max_sq2_states,
            steenrod_degree=certificate.sq2.degree,
        )
        return replay == certificate
    except (ValueError, TypeError, AttributeError, ArithmeticError):
        return False


def verify_symbolic_spherical(
    graph: nx.Graph,
    certificate: SymbolicIntegralMorseCertificate,
    *,
    max_vertices: int = 80,
    max_nodes: int = 250_000,
    max_critical: int = 128,
    max_gradient_states: int = 200_000,
    max_spherical_states: int = 2_000_000,
) -> bool:
    """Replay a complete spherical proof without searching for new spheres.

    Rebuild the matching, integral boundaries, connectivity witness and
    signed projections. Compare every selected cycle and lattice invariant;
    search work counters are not part of the mathematical claim.
    """
    try:
        proof = certificate.spherical
        if proof is None or not proof.complete:
            return False
        if certificate.wedge_spheres != certificate.betti or certificate.torsion:
            return False
        replay = symbolic_integral_morse_certificate(
            graph,
            vertex_order=certificate.vertex_order,
            max_vertices=max_vertices,
            max_nodes=max_nodes,
            max_critical=max_critical,
            max_gradient_states=max_gradient_states,
            max_spherical_states=max_spherical_states,
            spherical_seed=(c.pairs for d in proof.degrees for c in d.cycles),
            spherical_search=False,
        )
        if replay is None or replay.spherical is None:
            return False
        return (
            replay.spherical.complete
            and replay.spherical.degrees == proof.degrees
            and replay.critical == certificate.critical
            and replay.critical_masks == certificate.critical_masks
            and replay.boundaries == certificate.boundaries
            and replay.betti == certificate.betti
            and replay.torsion == certificate.torsion
            and replay.wedge_spheres == certificate.wedge_spheres
            and replay.connectivity_order == certificate.connectivity_order
            and replay.total_faces == certificate.total_faces
        )
    except (ValueError, TypeError, AttributeError, ArithmeticError):
        return False
