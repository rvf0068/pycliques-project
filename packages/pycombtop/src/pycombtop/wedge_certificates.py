"""Bounded exact wedge-of-spheres recognizers for flag complexes.

These graph-theoretic certificates use NO homology-to-homotopy inference.
Only elementary homotopy equivalences are accepted:

* A complete graph has contractible flag complex.
* An edgeless graph with n>=2 vertices has type wedge^(n-1) S^0.
* A connected triangle-free graph is a one-dimensional CW complex,
  homotopy equivalent to wedge^(m-n+1) S^1.
* The complement of k disjoint edges is the cross-polytope graph O_k,
  whose flag complex is S^(k-1) (k>=2).
* Disconnected complements give simplicial joins of independently
  certified factors, with the suspension/smash-product formula.
* At an articulation vertex, the flag complex is a wedge of the
  certified connected pieces.

Every recursive decomposition has strictly fewer vertices. A fixed
call budget fails closed, affecting only completeness, not soundness.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import networkx as nx

from .s_collapses import dismantling_certificate, verify_dismantling_certificate


@dataclass(frozen=True)
class FoldedWedgeCertificate:
    """Replayable vertex s-collapse and exact structural wedge recognition.

    Each move stores (removed vertex, dominating-vertex pairs for its open
    neighborhood). The final wedge type is checked again on the reduced
    graph; neither Betti numbers nor unchecked collapses are used.
    """

    spheres: dict[int, int]
    moves: tuple[tuple[Hashable, tuple[tuple[Hashable, Hashable], ...]], ...]


def verify_folded_wedge_certificate(
    graph: nx.Graph, certificate: FoldedWedgeCertificate
) -> bool:
    """Replay every local dismantling; reject malformed proofs, never raise."""
    try:
        current = graph.copy()
        for move in certificate.moves:
            if not isinstance(move, (tuple, list)) or len(move) != 2:
                return False
            vertex, local = move
            if vertex not in current:
                return False
            neighborhood = current.subgraph(list(current.neighbors(vertex))).copy()
            if not verify_dismantling_certificate(neighborhood, local):
                return False
            current.remove_node(vertex)
        return certified_flag_wedge(current) == certificate.spheres
    except (TypeError, ValueError):
        # The verifier is a trust boundary for potentially forged proofs.
        return False


def certified_flag_wedge_after_s_collapses(
    graph: nx.Graph,
    *,
    max_vertices: int = 18,
    max_checks: int = 48,
    max_moves: int = 4,
) -> FoldedWedgeCertificate | None:
    """Try a finite, replayable sequence of vertex s-collapses.

    First try the original graph with the existing exact wedge recognizer.
    Otherwise inspect at most max_checks vertex neighborhoods and make at
    most max_moves certified s-dismantlings, stopping as soon as a reduced
    graph has a recognizable wedge type. Enforce max_vertices before any
    s-collapse work; returning None means only that this *bounded* search
    did not find a certificate.

    A vertex s-dismantling preserves the simple-homotopy type of the
    flag complex because its open neighborhood is dismantlable.
    """
    direct = certified_flag_wedge(graph)
    if direct is not None:
        return FoldedWedgeCertificate(direct, ())
    if graph.number_of_nodes() > max_vertices or max_checks < 1 or max_moves < 1:
        return None
    current = graph.copy()
    moves: list[tuple[Hashable, tuple[tuple[Hashable, Hashable], ...]]] = []
    checked = 0
    for _ in range(max_moves):
        removed = False
        for vertex in tuple(current):
            if checked >= max_checks:
                return None
            checked += 1
            neighborhood = current.subgraph(list(current.neighbors(vertex))).copy()
            local = dismantling_certificate(neighborhood)
            if local is None:
                continue
            if not verify_dismantling_certificate(neighborhood, local):
                return None  # Fail closed if local proof cannot be replayed.
            moves.append((vertex, local))
            current.remove_node(vertex)
            removed = True
            spheres = certified_flag_wedge(current)
            if spheres is not None:
                certificate = FoldedWedgeCertificate(spheres, tuple(moves))
                if verify_folded_wedge_certificate(graph, certificate):
                    return certificate
                return None
            break
        if not removed:
            return None
    return None


def certified_flag_wedge(
    graph: nx.Graph, *, max_calls: int = 48
) -> dict[int, int] | None:
    """Return exactly certified sphere multiplicities, or None.

    The empty dictionary certifies contractibility. Dimension zero is
    admitted ONLY for edgeless graphs: a wedge of n-1 copies of S^0
    is the discrete space of n points, which is a valid *join factor*.
    Disconnected graphs with nontrivial connected components are
    deliberately not classified by this restricted recognizer.
    """
    if max_calls < 1:
        return None
    calls = 0

    def visit(g: nx.Graph) -> dict[int, int] | None:
        nonlocal calls
        calls += 1
        if calls > max_calls:
            return None
        n = int(g.number_of_nodes())
        if n == 0:
            return None
        e = int(g.number_of_edges())
        if e == n * (n - 1) // 2:
            return {}  # A simplex is contractible.
        if e == 0:
            return {0: n - 1}  # Discrete n-point space.
        if not nx.is_connected(g):
            return None

        complement = nx.complement(g)
        if (
            n >= 4
            and n % 2 == 0
            and all(degree == 1 for _, degree in complement.degree())
        ):
            return {n // 2 - 1: 1}  # Cross-polytope boundary.

        if not any(nx.triangles(g).values()):
            # Every connected graph is a one-dimensional CW complex
            # when triangle-free, and has free fundamental group.
            rank = e - n + 1
            return {1: rank} if rank else {}

        components = list(nx.connected_components(complement))
        if len(components) > 1:
            factors: dict[int, int] | None = None
            for component in components:
                factor = nx.complement(complement.subgraph(component).copy())
                factor_type = visit(factor)
                if factor_type is None:
                    return None
                if not factor_type:
                    return {}  # Joining with a cone produces a cone.
                if factors is None:
                    factors = factor_type
                else:
                    joined: dict[int, int] = {}
                    for i, a in factors.items():
                        for j, b in factor_type.items():
                            dim = i + j + 1
                            joined[dim] = joined.get(dim, 0) + a * b
                    factors = joined
            return factors

        articulation = next(nx.articulation_points(g), None)
        if articulation is not None:
            deletion = g.subgraph([v for v in g if v != articulation])
            spheres: dict[int, int] = {}
            for component in nx.connected_components(deletion):
                piece = g.subgraph(set(component) | {articulation})
                wedge = visit(piece)
                if wedge is None or 0 in wedge:
                    return None
                for d, multiplicity in wedge.items():
                    spheres[d] = spheres.get(d, 0) + multiplicity
            return spheres

        return None

    return visit(graph)
