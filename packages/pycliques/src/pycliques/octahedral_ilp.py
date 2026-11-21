"""Optional mixed-integer search for induced octahedra in finite simple graphs.

An induced O_d in G is an induced matching of size d in complement(G).
The MILP uses one binary variable per complement edge and one per vertex.
It never relies on predefined vertices and never treats a search failure
as a proof of K-convergence or of the absence of a retraction.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from math import isfinite

import networkx as nx


class OctahedronILPStatus(Enum):
    """Exact outcomes and bounded/inconclusive outcomes."""

    OPTIMAL = "optimal"
    FOUND = "found"
    FEASIBLE_LIMIT = "feasible_limit"
    INFEASIBLE = "infeasible"
    LIMIT_REACHED = "limit_reached"
    SOLVER_ERROR = "solver_error"


@dataclass(frozen=True)
class OctahedronILPResult:
    """Induced antipodal pairs, if any, and the MILP solver's status.

    OPTIMAL means the MILP solver reported an optimum for the selected
    objective (maximum dimension if requested). FOUND means the solver
    found a feasible target without an optimization objective.
    FEASIBLE_LIMIT contains a verified incumbent but not an optimum.
    INFEASIBLE means the formulation was proved infeasible by the solver
    or by an immediate necessary size/degree condition.
    LIMIT_REACHED and SOLVER_ERROR never certify a negative result.
    None of these statuses, including OPTIMAL, establishes a retraction.
    """

    status: OctahedronILPStatus
    antipodal_pairs: tuple[tuple[Hashable, Hashable], ...] | None
    dimension: int | None
    message: str


def _valid_target(
    graph: nx.Graph,
    pairs: tuple[tuple[Hashable, Hashable], ...],
    *,
    require_dominating: bool,
) -> bool:
    """Verify all pairs and all target cross-edges without trusting MILP."""
    chosen = [x for pair in pairs for x in pair]
    if len(set(chosen)) != len(chosen):
        return False
    for i, (u, v) in enumerate(pairs):
        if graph.has_edge(u, v):
            return False
        for a, b in pairs[:i]:
            if not all(graph.has_edge(x, y) for x in (u, v) for y in (a, b)):
                return False
    if require_dominating:
        endpoints = set(chosen)
        if any(all(v == u or graph.has_edge(v, u) for u in endpoints) for v in graph):
            return False
    return True


def find_induced_octahedron_ilp(
    graph: nx.Graph,
    *,
    min_dimension: int = 3,
    maximize_dimension: bool = True,
    require_dominating: bool = True,
    time_limit: float | None = None,
) -> OctahedronILPResult:
    """Find an induced O_d using an optional SciPy/HiGHS integer solver.

    Let H = complement(G), x_uv indicate a selected edge of H and y_v
    indicate a selected endpoint. The integer constraints are:

    * y_v = sum(x_uv : uv in E(H)) for every vertex v (a matching),
    * y_u + y_v - x_uv <= 1 for each uv in E(H) (induced matching),
    * sum(y_u : u in N_H(v)) >= 1 for all v, when domination requested,
    * sum(x_uv) >= min_dimension.

    Maximize sum(x_uv) when maximize_dimension is True, otherwise seek any
    feasible point. The optional time_limit is passed to HiGHS.
    A feasible incumbent obtained at a time limit is validated but is
    *not* claimed to be optimal. The search never tests for retractions.

    SciPy is loaded only on invocation; pip install scipy enables it.
    Do not use an unsuccessful bounded run as a negative certificate.
    """
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
    ):
        raise ValueError("graph must be a simple undirected loopless NetworkX Graph")
    if type(min_dimension) is not int or min_dimension < 2:
        raise ValueError("min_dimension must be an integer >= 2")
    if type(maximize_dimension) is not bool:
        raise ValueError("maximize_dimension must be bool")
    if type(require_dominating) is not bool:
        raise ValueError("require_dominating must be bool")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and strictly positive")

    nodes = list(graph)
    n = len(nodes)
    impossible = OctahedronILPResult(
        OctahedronILPStatus.INFEASIBLE,
        None,
        None,
        "No qualifying induced octahedron exists (necessary condition).",
    )
    if n < 2 * min_dimension:
        return impossible

    position = {v: i for i, v in enumerate(nodes)}
    edges = [(position[u], position[v]) for u, v in nx.non_edges(graph)]
    m = len(edges)
    if m < min_dimension:
        return impossible

    incident: list[list[int]] = [[] for _ in nodes]
    neighbors: list[list[int]] = [[] for _ in nodes]
    for k, (u, v) in enumerate(edges):
        incident[u].append(k)
        incident[v].append(k)
        neighbors[u].append(v)
        neighbors[v].append(u)
    if require_dominating and any(not nbrs for nbrs in neighbors):
        # A universal vertex of G would be adjacent to every candidate.
        return impossible

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for the optional induced-octahedron ILP search; "
            "install it with pip install scipy"
        ) from exc

    # Columns [x_e for complement edges] + [y_v for graph vertices].
    # Build sparse constraints directly, avoiding a quadratic conflict graph.
    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    def add_row(entries: list[tuple[int, float]], low: float, high: float) -> None:
        r = len(lb)
        for c, val in entries:
            row_ids.append(r)
            col_ids.append(c)
            values.append(val)
        lb.append(low)
        ub.append(high)

    for v, contained in enumerate(incident):
        add_row([(m + v, 1.0)] + [(k, -1.0) for k in contained], 0.0, 0.0)
    for k, (u, v) in enumerate(edges):
        add_row([(m + u, 1.0), (m + v, 1.0), (k, -1.0)], -np.inf, 1.0)
    if require_dominating:
        for nbrs in neighbors:
            add_row([(m + v, 1.0) for v in nbrs], 1.0, np.inf)
    add_row([(k, 1.0) for k in range(m)], float(min_dimension), np.inf)

    A = coo_matrix(
        (np.asarray(values), (np.asarray(row_ids), np.asarray(col_ids))),
        shape=(len(lb), m + n),
    ).tocsr()
    c = np.zeros(m + n)
    if maximize_dimension:
        c[:m] = -1.0
    options: dict[str, float | bool] = {"mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    try:
        result = milp(
            c,
            integrality=np.ones(m + n),
            bounds=Bounds(np.zeros(m + n), np.ones(m + n)),
            constraints=LinearConstraint(A, np.asarray(lb), np.asarray(ub)),
            options=options,
        )
    except Exception as exc:
        return OctahedronILPResult(
            OctahedronILPStatus.SOLVER_ERROR, None, None, f"Solver error: {exc}"
        )

    details = str(result.message)
    if result.status == 2:
        return OctahedronILPResult(OctahedronILPStatus.INFEASIBLE, None, None, details)

    if result.status in (0, 1) and result.x is not None:
        selected = np.asarray(result.x[:m], dtype=float)
        if np.all(np.isfinite(selected)) and np.all(
            np.abs(selected - np.rint(selected)) < 1e-5
        ):
            pairs = tuple(
                (nodes[u], nodes[v])
                for k, (u, v) in enumerate(edges)
                if selected[k] > 0.5
            )
            if len(pairs) >= min_dimension and _valid_target(
                graph, pairs, require_dominating=require_dominating
            ):
                status = (
                    OctahedronILPStatus.FOUND
                    if not maximize_dimension
                    else (
                        OctahedronILPStatus.FEASIBLE_LIMIT
                        if result.status == 1
                        else OctahedronILPStatus.OPTIMAL
                    )
                )
                return OctahedronILPResult(status, pairs, len(pairs), details)
        return OctahedronILPResult(
            OctahedronILPStatus.SOLVER_ERROR,
            None,
            None,
            "The proposed incumbent failed independent verification: " + details,
        )

    return OctahedronILPResult(
        OctahedronILPStatus.LIMIT_REACHED
        if result.status == 1
        else OctahedronILPStatus.SOLVER_ERROR,
        None,
        None,
        details,
    )
