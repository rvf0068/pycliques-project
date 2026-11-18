"""Certified low-dimensional Hopf attaching maps via integral Morse and cup products.

For a connected flag complex whose augmented acyclic Morse model has one
2-cell, no essential 1- or 3-cells, and s 4-cells, the surviving 4-cell
attachments lie in pi_3(S^2) = Z. Their Hopf invariants are the coefficients
of the cup square of a generator of H^2. This module proves that square
vanishes by finding a maximal triangle whose indicator is a nonzero
mod-2 cohomology class: it is an *integral* cocycle, and its cup square
vanishes identically as a cochain. The Smith hypotheses guarantee that
H^4 is torsion-free; therefore the integral generator also squares to 0.

This method does not promote Betti numbers to homotopy equivalences.
Failure to satisfy every check is inconclusive. The dense reduction
wrapper is opt-in and does not change default census heuristics.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import networkx as nx
from sympy import ZZ, Matrix
from sympy.matrices.normalforms import smith_normal_form

from .flag_edge_star import (
    FlagReductionCertificate,
    _canonical_core,
    reduce_flag_graph,
    verify_flag_reduction,
)
from .simplex import _stable_vertex_key
from .symbolic_integral_morse import (
    SymbolicIntegralMorseCertificate,
    symbolic_integral_morse_certificate,
)


@dataclass(frozen=True)
class HopfWedgeCertificate:
    """Replayable proof of a wedge S^2 vee k*S^4.

    triangle_indices uses the same stable vertex ordering as symbolic Dong,
    not external graph labels. ``boundaries`` are signed integral cellular
    Morse matrices. A verifier recomputes all evidence.
    """

    triangle_indices: tuple[int, int, int]
    critical: dict[int, int]
    boundaries: dict[int, tuple[tuple[int, ...], ...]]
    betti: dict[int, int]
    spheres: dict[int, int]
    triangle_count: int


@dataclass(frozen=True)
class ReducedHopfWedgeCertificate:
    """Homotopy-preserving graph reduction followed by a Hopf wedge proof."""

    reduction: FlagReductionCertificate
    hopf: HopfWedgeCertificate
    spheres: dict[int, int]


def _ordered_vertices(graph: nx.Graph) -> list[Hashable]:
    if all(isinstance(v, int) for v in graph):
        return sorted(graph)
    return sorted(graph, key=_stable_vertex_key)


def _unit_boundary(
    boundary: tuple[tuple[int, ...], ...] | None, rows: int, cols: int
) -> bool:
    """Require a surjective integer differential with unit Smith invariants."""
    if rows == 0:
        return True
    if boundary is None or len(boundary) != rows:
        return False
    if any(len(row) != cols for row in boundary):
        return False
    normal = smith_normal_form(Matrix(boundary), domain=ZZ)
    return all(abs(int(normal[i, i])) == 1 for i in range(rows))


def _hopf_shape(
    morse: SymbolicIntegralMorseCertificate,
) -> dict[int, int] | None:
    """Recognize precisely the CW shape to which the Hopf criterion applies."""
    critical = morse.critical
    if (
        not critical
        or set(critical) - {0, 1, 2, 3, 4}
        or critical.get(2) != 1
        or critical.get(4, 0) < 1
    ):
        return None
    c0, c1 = critical.get(0, 0), critical.get(1, 0)
    if (c0, c1) not in ((0, 0), (1, 1)):
        return None
    if (c0, c1) == (1, 1):
        if not _unit_boundary(morse.boundaries.get(1), 1, 1):
            return None
    if morse.boundaries.get(2, ((0,),)) not in ((), ((0,),)):
        return None
    n3 = critical.get(3, 0)
    if n3 and morse.boundaries.get(3) != (tuple(0 for _ in range(n3)),):
        return None
    n4 = critical[4]
    if n3 and not _unit_boundary(morse.boundaries.get(4), n3, n4):
        return None
    if morse.torsion:
        return None
    spheres = {2: 1, 4: n4 - n3}
    if spheres[4] < 0:
        return None
    if {d: n for d, n in spheres.items() if n} != morse.betti:
        return None
    return {d: n for d, n in spheres.items() if n}


def _mod2_coboundary_basis(
    edges: list[tuple[int, int]], triangles: list[tuple[int, int, int]]
) -> dict[int, int]:
    """Gaussian elimination of C^1 -> C^2 over F2 using integer bitsets."""
    edge_id = {edge: i for i, edge in enumerate(edges)}
    columns = [0] * len(edges)
    for t, (a, b, c) in enumerate(triangles):
        for edge in ((a, b), (a, c), (b, c)):
            columns[edge_id[edge]] |= 1 << t
    basis: dict[int, int] = {}
    for col in columns:
        while col:
            pivot = col.bit_length() - 1
            if pivot in basis:
                col ^= basis[pivot]
            else:
                basis[pivot] = col
                break
    return basis


def _not_coboundary(basis: dict[int, int], target: int) -> bool:
    while target:
        pivot = target.bit_length() - 1
        if pivot not in basis:
            return True
        target ^= basis[pivot]
    return False


def _cup_witness(
    graph: nx.Graph,
    morse: SymbolicIntegralMorseCertificate,
    *,
    max_triangles: int,
    preferred: Iterable[tuple[int, int, int]] = (),
) -> tuple[tuple[int, int, int], int] | None:
    """Find a maximal triangle representing an odd H^2 generator.

    An indicator of a maximal triangle is an integral 2-cocycle. Its
    Alexander-Whitney self-cup vanishes *on every 4-simplex*: the first
    and last three ordered vertices cannot denote the same triangle.
    Nonzero mod-2 cohomology means the integral class is an odd multiple
    of the generator of H^2=Z, and torsion-free H^4 makes the vanishing
    integral cup square descend to the generator itself.
    """
    if max_triangles < 1:
        return None
    indexed = _canonical_core(graph)
    if not indexed:
        return None
    triangles: list[tuple[int, int, int]] = []
    for face in nx.enumerate_all_cliques(indexed):
        if len(face) > 3:
            break
        if len(face) == 3:
            triangles.append(tuple(sorted(face)))
            if len(triangles) > max_triangles:
                return None
    if not triangles:
        return None
    triangle_id = {triangle: i for i, triangle in enumerate(triangles)}
    edges = sorted(tuple(sorted(e)) for e in indexed.edges)
    basis = _mod2_coboundary_basis(edges, triangles)

    critical_triangles = [
        tuple(i for i in range(len(indexed)) if mask >> i & 1)
        for mask in morse.critical_masks
        if mask.bit_count() == 3
    ]
    candidates = dict.fromkeys((*preferred, *critical_triangles, *triangles))
    for triangle in candidates:
        t = triangle_id.get(triangle)
        if t is None:
            continue
        a, b, c = triangle
        if set(indexed[a]) & set(indexed[b]) & set(indexed[c]):
            continue
        if _not_coboundary(basis, 1 << t):
            return triangle, len(triangles)
    return None


def _from_morse(
    graph: nx.Graph,
    morse: SymbolicIntegralMorseCertificate,
    *,
    max_triangles: int,
    preferred: Iterable[tuple[int, int, int]] = (),
) -> HopfWedgeCertificate | None:
    spheres = _hopf_shape(morse)
    if spheres is None or not nx.is_connected(graph):
        return None
    witness = _cup_witness(
        graph, morse, max_triangles=max_triangles, preferred=preferred
    )
    if witness is None:
        return None
    triangle, count = witness
    return HopfWedgeCertificate(
        triangle, morse.critical, morse.boundaries, morse.betti, spheres, count
    )


def hopf_wedge_certificate(
    graph: nx.Graph,
    *,
    max_vertices: int = 80,
    max_triangles: int = 20_000,
    max_nodes: int = 250_000,
    max_gradient_states: int = 200_000,
) -> HopfWedgeCertificate | None:
    """Certify S^2 vee k*S^4, if all Morse and cup-product tests pass."""
    if (
        not graph
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
        or len(graph) > max_vertices
        or max_triangles < 1
    ):
        return None
    morse = symbolic_integral_morse_certificate(
        graph,
        max_vertices=max_vertices,
        max_nodes=max_nodes,
        max_gradient_states=max_gradient_states,
    )
    if morse is None:
        return None
    proof = _from_morse(graph, morse, max_triangles=max_triangles)
    return proof if proof is not None and verify_hopf_wedge(graph, proof) else None


def verify_hopf_wedge(
    graph: nx.Graph,
    proof: HopfWedgeCertificate,
    *,
    max_triangles: int = 20_000,
) -> bool:
    """Independently recompute both exact integral and mod-2 certificates."""
    try:
        if (
            not graph
            or graph.is_directed()
            or graph.is_multigraph()
            or nx.number_of_selfloops(graph)
            or len(graph) > 80
        ):
            return False
        morse = symbolic_integral_morse_certificate(graph)
        if morse is None:
            return False
        expected = _from_morse(
            graph,
            morse,
            max_triangles=max_triangles,
            preferred=(proof.triangle_indices,),
        )
        return (
            expected is not None
            and expected.triangle_indices == proof.triangle_indices
            and expected.triangle_count == proof.triangle_count
            and expected.critical == proof.critical
            and expected.boundaries == proof.boundaries
            and expected.betti == proof.betti
            and expected.spheres == proof.spheres
        )
    except (ArithmeticError, KeyError, TypeError, ValueError, MemoryError, RecursionError):
        return False


def reduced_hopf_wedge_certificate(
    graph: nx.Graph,
    *,
    max_vertices: int = 1000,
    **reduction_options,
) -> ReducedHopfWedgeCertificate | None:
    """Opt-in certified edge reductions followed by the Hopf criterion."""
    if not graph or len(graph) > max_vertices:
        return None
    reduced = reduce_flag_graph(graph, max_vertices=max_vertices, **reduction_options)
    if reduced is None:
        return None
    core, reduction = reduced
    proof = hopf_wedge_certificate(_canonical_core(core))
    if proof is None:
        return None
    combined = ReducedHopfWedgeCertificate(reduction, proof, proof.spheres)
    return combined if verify_reduced_hopf_wedge(graph, combined) else None


def verify_reduced_hopf_wedge(
    graph: nx.Graph, proof: ReducedHopfWedgeCertificate
) -> bool:
    """Replay the entire homotopy-preserving reduction and Hopf obstruction."""
    if not verify_flag_reduction(graph, proof.reduction):
        return False
    core = nx.Graph()
    core.add_nodes_from(proof.reduction.remaining_vertices)
    core.add_edges_from(proof.reduction.remaining_edges)
    return (
        verify_hopf_wedge(_canonical_core(core), proof.hopf)
        and proof.spheres == proof.hopf.spheres
    )
