# Optional mixed-integer search for induced octahedra

The independent module `pycliques.octahedral_ilp` finds induced
`O_d = complement(d K_2)` targets in an arbitrary finite simple graph.
It uses SciPy/HiGHS, rather than randomized compatibility or amplification.
No graph-specific embeddings, target labels, or saved retraction maps
occur anywhere in the algorithm. It **does not test retractions**.

This research branch deliberately leaves the public `pycliques.__init__`,
the compatibility-search module of PR #31, and the existing algorithms
untouched. Install SciPy separately (it is imported lazily):

```sh
uv pip install 'scipy>=1.11'
```

## Usage

```python
import networkx as nx
from pycliques.octahedral_ilp import (
    OctahedronILPStatus, find_induced_octahedron_ilp
)

G = nx.complete_multipartite_graph(2, 2, 2, 2)
result = find_induced_octahedron_ilp(
    G, min_dimension=3,
    maximize_dimension=True,
    require_dominating=True,
    time_limit=30,
)
assert result.status is OctahedronILPStatus.OPTIMAL
assert result.dimension == 4
assert result.antipodal_pairs is not None
```

The returned `antipodal_pairs` use the original NetworkX vertex
labels; each reported target is independently checked against the input.

| Status | Meaning |
| --- | --- |
| `OPTIMAL` | HiGHS reports optimal dimension, and a target is verified |
| `FOUND` | Feasibility-only solve produced a verified target, even with a time limit |
| `FEASIBLE_LIMIT` | Optimization stopped with a verified incumbent; maximum dimension is **unknown** |
| `INFEASIBLE` | No induced target satisfying the requested constraints exists |
| `LIMIT_REACHED` | No incumbent within the resource budget: **inconclusive** |
| `SOLVER_ERROR` | Solver failed or produced an invalid candidate: **inconclusive** |

An `OPTIMAL` status refers to the chosen domination constraint: when
`require_dominating=True` it maximizes dimension only among the
complement-dominating targets. MILP optimality is solver-reported under
numeric tolerances, not a separately checked exact-rational proof.
Existence of the returned target is independently checked.

**An induced octahedron is not automatically a retract.** A candidate
must additionally pass `pycliques.find_octahedral_retraction`, with a
complete verified projection, before an octahedral-retraction divergence
theorem can be invoked. Maximizing dimension can find better targets,
but a larger target need not admit a retraction.

## The linear constraints

Let `H = complement(G)`. An induced `O_d` in `G` is an induced
matching of size `d` in `H`. Introduce binary `x_uv` for each
edge `uv` of `H`, and binary `y_v` for each vertex. Impose

\[
\begin{aligned}
y_v &= \sum_{e\ni v}x_e && (v\in V(H)),\\
y_u+y_v-x_{uv} &\le 1 && (uv\in E(H)),\\
\sum_{u\in N_H(v)}y_u &\ge 1 &&
    (v\in V(H),\text{ if domination required}),\\
\sum_{e\in E(H)}x_e &\ge d_{\min}.
\end{aligned}
\]

The first row ensures the selected edges are a matching; the second
forbids an edge of `H` joining endpoints of distinct matching edges.
The optional third is total domination of `H` by matched endpoints,
equivalent to the empty common closed-neighborhood intersection of the
selected vertices in `G`. Such domination is necessary for a retraction
onto the induced octahedron: an external vertex adjacent to all target
vertices cannot be mapped into that target while fixing it pointwise.

The objective is to maximize `sum(x_e)` if `maximize_dimension=True`;
otherwise it is pure feasibility. The sparse formulation has
`|V(H)| + |E(H)|` binary variables and at most
`2|V(H)| + |E(H)| + 1` constraints. Its worst-case time is exponential.

## Census regression

For the completely pared fourth iterate of McKay **10510367**
(original graph6 `ICZVByz~_`), a 64-vertex graph with 1,844 edges
archived in [census PR #60](https://github.com/rvf0068/pycliques-census/pull/60),
HiGHS finds a complement-dominating induced **O12** and reports maximum
dimension 12. The test embeds that archived P4 graph6 as a *fixture*,
not as a hardcoded input to the finder. The resulting O12 may be
different from the target of the previously certified retraction.

The existing 1,076-vertex fourth snub iterate and blind O20-to-O24
work remain separate: randomized compatibility search may be preferable
on such large inputs.

To reproduce, in the project workspace with optional SciPy installed:

```sh
uv run --no-sync pytest packages/pycliques/tests/test_octahedral_ilp.py -m 'not slow'
uv run --no-sync pytest packages/pycliques/tests/test_octahedral_ilp.py -m slow
```
