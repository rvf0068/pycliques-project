"""Verified integer-weight lower bounds for connectivity of flag complexes.

If nonnegative integer weights on graph vertices sum to W and each closed
non-neighborhood has weight at most M, then any r vertices admit a common
neighbor whenever r*M < W. The clique complex is therefore r-conic, and
Barmak's theorem gives (floor(r/2)-1)-connectivity. Optimization is merely
an optional source of candidate weights; the verifier uses exact integers.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import networkx as nx

from .simplex import _stable_vertex_key


@dataclass(frozen=True)
class WeightedConicityCertificate:
    """Exact weight witness for conicity and a connectivity lower bound.

    ``weights`` is a tuple aligned with ``vertices``. A nonzero bound ``r``
    asserts every set of up to r vertices has an external common neighbor.
    Connectivity is interpreted as at least -1 for a nonempty complex.
    """

    vertices: tuple[Hashable, ...]
    weights: tuple[int, ...]
    total_weight: int
    max_closed_non_neighbor_weight: int
    conicity: int
    connectivity: int


def verify_weighted_conicity(
    graph: nx.Graph, weights: Mapping[Hashable, int]
) -> WeightedConicityCertificate:
    """Verify a proposed conicity weight certificate with integer arithmetic.

    The graph is interpreted via its clique complex. All weights must be
    nonnegative integers, with positive total weight and exactly the graph's
    vertices as keys. Does not require SciPy.
    """
    if set(weights) != set(graph):
        raise ValueError("weights must have exactly the graph vertices as keys")
    vertices = tuple(sorted(graph, key=_stable_vertex_key))
    if any(
        isinstance(weights[v], bool) or not isinstance(weights[v], int)
        for v in vertices
    ):
        raise ValueError("weights must be nonnegative integers")
    values = tuple(weights[v] for v in vertices)
    if any(x < 0 for x in values):
        raise ValueError("weights must be nonnegative integers")
    total = sum(values)
    if total <= 0:
        raise ValueError("total weight must be positive")
    # N_complement[v] = V(graph) \ N_graph(v).
    max_closed = max(total - sum(weights[u] for u in graph[v]) for v in vertices)
    if max_closed <= 0:
        raise ArithmeticError("nonpositive closed non-neighborhood weight")
    r = (total - 1) // max_closed
    return WeightedConicityCertificate(
        vertices=vertices,
        weights=values,
        total_weight=total,
        max_closed_non_neighbor_weight=max_closed,
        conicity=r,
        connectivity=r // 2 - 1,
    )


def find_weighted_conicity_certificate(
    graph: nx.Graph,
    *,
    max_vertices: int = 1000,
    time_limit: float = 15.0,
    weight_scale: int = 10_000_000,
) -> WeightedConicityCertificate | None:
    """Find and *verify* nonnegative integer weights giving a conicity bound.

    The linear program minimizes the maximum closed non-neighborhood mass
    under sum(weights)=1. A rounded integer solution is always independently
    verified, and accepted only if it improves upon uniform weights.
    Returns a uniform certificate if optimization fails, or None when the
    order exceeds max_vertices. Uses SciPy's HiGHS solver.
    """
    if max_vertices < 1 or weight_scale < 1 or time_limit <= 0:
        raise ValueError("resource limits must be positive")
    n = len(graph)
    if n == 0 or n > max_vertices:
        return None
    vertices = list(sorted(graph, key=_stable_vertex_key))
    uniform = verify_weighted_conicity(graph, dict.fromkeys(vertices, 1))
    if n == 1:
        return uniform

    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, csr_matrix, hstack

    index = {v: i for i, v in enumerate(vertices)}
    rows: list[int] = []
    cols: list[int] = []
    for i, v in enumerate(vertices):
        neighbors = set(graph[v])
        for u in vertices:
            if u not in neighbors:
                rows.append(i)
                cols.append(index[u])
    matrix = coo_matrix(
        (np.ones(len(rows), dtype=float), (rows, cols)), shape=(n, n)
    ).tocsr()
    # Each complement closed neighborhood has weight <= t.
    inequalities = hstack([matrix, csr_matrix(-np.ones((n, 1)))], format="csr")
    equalities = csr_matrix(
        (np.ones(n), (np.zeros(n, dtype=int), np.arange(n))), shape=(1, n + 1)
    )
    objective = np.zeros(n + 1)
    objective[-1] = 1.0
    try:
        outcome = linprog(
            objective,
            A_ub=inequalities,
            b_ub=np.zeros(n),
            A_eq=equalities,
            b_eq=np.array([1.0]),
            bounds=[(0, None)] * (n + 1),
            method="highs",
            options={"time_limit": float(time_limit)},
        )
        if not outcome.success or outcome.x is None:
            return uniform
        positive = np.maximum(outcome.x[:n], 0.0)
        if not np.isfinite(positive).all() or positive.sum() <= 0:
            return uniform
        positive /= positive.sum()
        rounded = [int(round(x * weight_scale)) for x in positive]
        if sum(rounded) == 0:
            return uniform
        proposed = verify_weighted_conicity(graph, dict(zip(vertices, rounded)))
        return proposed if proposed.conicity > uniform.conicity else uniform
    except (ArithmeticError, MemoryError, ValueError):
        return uniform
