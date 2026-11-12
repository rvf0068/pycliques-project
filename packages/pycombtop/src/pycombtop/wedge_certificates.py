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

import networkx as nx


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
