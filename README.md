[![CI](https://github.com/rvf0068/pycliques-project/actions/workflows/ci.yml/badge.svg)](https://github.com/rvf0068/pycliques-project/actions/workflows/ci.yml) [![Lint](https://github.com/rvf0068/pycliques-project/actions/workflows/lint.yml/badge.svg)](https://github.com/rvf0068/pycliques-project/actions/workflows/lint.yml) [![codecov](https://codecov.io/gh/rvf0068/pycliques-project/branch/main/graph/badge.svg)](https://codecov.io/gh/rvf0068/pycliques-project)

# pycliques-project

A monorepo for a graph-theory and combinatorial-topology ecosystem built on [NetworkX](https://networkx.org/). The workspace contains four packages:

| Package | Description |
|---|---|
| [`pycliques`](#pycliques) | Clique graph operator K(G) |
| [`pyg6data`](#pyg6data) | Graph datasets in graph6 format |
| [`pycombtop`](#pycombtop) | Combinatorial topology and simplicial complexes |
| [`pyhomrep`](#pyhomrep) | Group representations on simplicial homology |

## Installation

All packages live under `packages/` in this repository. Clone and install the ones you need:

```bash
git clone https://github.com/rvf0068/pycliques-project.git
cd pycliques-project
```

**With uv** (recommended — resolves the sibling packages from this checkout):

```bash
uv sync                        # set up the cloned workspace (all packages + dev tools)
```

Use `uv add pycliques` only to add the published package to a *different*
project; it edits that project's dependencies and is not a setup step here.

**With pip** (install individual packages from this checkout). Install
`pyg6data` before packages that depend on it, and use local paths so pip does
not resolve sibling packages from PyPI:

```bash
pip install ./packages/pyg6data
pip install ./packages/pycliques
pip install ./packages/pycombtop
pip install ./packages/pyhomrep
```

Mathematical sources for the implemented theorems are listed in
[`docs/references.rst`](docs/references.rst) (built into the Sphinx docs). They
justify the theorems and algorithms, not the experimental census outcomes.

---

## pycliques

**Terminology:** In this project, a *complete* is a mutually adjacent set of
vertices, and a *clique* is a maximal complete. This follows the clique-graph
literature; in particular, it differs from the common modern usage in which
"clique" may mean any complete set.

Tools for studying the **clique graph operator** K(G). Given a graph G, the clique graph K(G) has the cliques of G as vertices, with two cliques adjacent when they share a vertex. The package provides:

- `clique_graph` — compute K(G) with an optional clique-count bound
- `is_clique_helly` / `is_hereditary_clique_helly` — Helly property tests
- `completely_pared_graph` / `find_dominated_vertex` — graph dismantling
- `recognize_clockwork` / `is_clique_divergent_clockwork` — clockwork-graph recognition
- `retracts` / `special_octahedra_dimension` — retraction tests
- `classify_clique_behavior` — apply the fast standard convergence/divergence
  tests to one graph
- `classify_clockwork_pair_map` — the separate, more expensive clockwork-pair
  (coaffination) search based on [LNP06, Theorems 2.6 and 3.1]
- `classify_clique_behavior_with_clockwork_pair_search` — runs the fast pass, then
  the clockwork-pair search only if the fast pass left the graph `INDETERMINATE`
- CLI `small-behavior` — fast classification of all connected graphs of a
  given order (never runs the clockwork-pair search)
- CLI `small-behavior-clockwork-pairs` — optional, more expensive second pass:
  applies the clockwork-pair search only to graphs the fast pass left `INDETERMINATE`

**Dependencies:** `networkx`, `grandiso`, `rich`  
**Optional:** `pyg6data` (needed only for the `small-behavior` /
`small-behavior-clockwork-pairs` CLIs — install with
`pip install "./packages/pycliques[data]"`)

### Example

```python
import networkx as nx
from pycliques import Verdict, classify_clique_behavior

g = nx.octahedral_graph()
result = classify_clique_behavior(g, tries=9, bound=30)
print(result.verdict is Verdict.DIVERGENT)  # True
print(result.reason)  # eventually has a special octahedron (index 0, dimension 3)
```

`INDETERMINATE` means that the available sufficient tests did not decide the
graph within the requested iteration and clique-count limits. Check
`result.bound_exceeded` to distinguish a clique-count abort from an
indeterminate result after all requested iterations.

### Suspension fast criterion ([LNP09, Theorem 4.6])

The fast classifier also applies a special case of Neumann-Lara's connected
summand theorem, [LNP09, Theorem 4.6] (not its verbatim full statement): if
`S = H * complement(K_2)`, where `H` is connected and admits a 2-coaffination,
then `S` is expansive and therefore clique divergent. The implementation looks
for two nonadjacent universal vertices whose deletion leaves a connected base
graph, then reuses the general `coaffinations(H, 2)` machinery. It does not
search arbitrary induced subgraphs or require the coaffination to be an
involution. A clique is understood here as a maximal complete subgraph.

### Two-level classification: fast pass vs. clockwork-pair search

`classify_clique_behavior` (and the `small-behavior` CLI) is the **fast
pass**: it applies complete paring, direct reference graphs, clockwork
criteria, eventual-Helly and special-octahedron tests, fixed retractions,
reference dependency, [FANP13, Theorems 6.1/6.2] local-bridge and non-triangle-edge
inference, and inverse cutpoint extension. It deliberately excludes the
expensive clockwork-pair (rank-divergence, [LNP06, Theorems 2.6 and 3.1]) coaffination search, so it stays fast
even for large censuses.

The clockwork-pair search is available as a separate, optional second stage
(`classify_clockwork_pair_map` / the `small-behavior-clockwork-pairs` CLI). It
searches, in increasing radius order (`r = 3, 4, 5, ...`, i.e. `m = 2, 3, 4,
...` with `r = m + 1`), for an admissible morphism of coaffine pairs from a
clockwork graph `R_{2m}^n` into the target graph -- or, failing that, into
its clique graph `K(G)` (one fallback level only, bounded by the same
`--bound`/clique-count limit). Because it may enumerate many candidate
target coaffinations and search several `(m, n)` combinations, this pass can
be substantially more expensive than the fast pass, so it is run only on
graphs the fast pass left `INDETERMINATE`:

```bash
uv run small-behavior 9                       # fast pass; saves INDETERMINATE graphs
uv run small-behavior 9 --from-indeterminate-file --bound 60
                                                # recheck saved graphs with a larger bound
uv run small-behavior 9 --from-indeterminate-file \
  --exclude-conjectured-divergent
                        # leave conjectured-divergent rows untouched
uv run small-behavior 9 --from-indeterminate-file \
  --check-clique-retraction
                        # run the seq[1] Comp(C_10) retraction pass
uv run small-behavior-clockwork-pairs 9 --max-m 3   # clockwork-pair pass on the unresolved graphs
```

The fast CLI's `--from-indeterminate-file` mode reads the saved
`indeterminate_order_<n>.txt` file, applies the default fast-pass tests again,
and rewrites the file with only graphs that remain `INDETERMINATE`. Use
`--no-save` to inspect the results without updating the file.
With `--exclude-conjectured-divergent`, rows carrying a saved
`conjectured_divergent` certificate are preserved without being re-tested.
With `--check-clique-retraction`, the pass runs
`_make_clique_retraction_test(complement_of_cycle(10), ...)` on each saved
graph and removes rows whose clique graph retracts to `Comp(C_10)`. Each
studied row is logged at INFO level with its original index and graph6 string.

On a fresh census run, each CLI updates only original indices it actually
processed: stale unresolved rows for those indices are removed, newly
unresolved rows are written, and records outside a selected `--start`/`--end`
range are preserved. Graphs skipped as reducible are included in the processed
cohort. The fast CLI gives the current certified verdict precedence over any
isomorphic unresolved core found by lookup; a prior-core match only annotates
a graph that is still indeterminate.

For `small-behavior-clockwork-pairs --from-indeterminate-file`, the saved file is the
entire input cohort and is replaced with exactly the rows still unresolved
after the pass, including a header-only file when all candidates resolve.
`--start` and `--end` are rejected in this mode because saved-file entries do
not define a census interval. `--no-save` leaves the unresolved file unchanged;
`--output-file` can still record per-run verdicts. In fresh clockwork-pair runs,
only indices actually visited by the fast pass are updated, preserving saved
rows outside a partial range.

The `small-behavior-clockwork-pairs` CLI exposes `--max-m`, `--max-n`,
`--max-coaffinations`, `--max-source-order`, and `--bound` so that
progressively larger computational limits can be tried without modifying
source code, and `--from-indeterminate-file` to reuse a previously saved
`indeterminate_order_<n>.txt` file instead of rerunning the fast pass.

---

## pyg6data

Access to [Brendan McKay's](http://cs.anu.edu.au/~bdm/data/graphs.html) graph census and the [House of Graphs](https://houseofgraphs.org/) cubic-graph database, bundled as compressed graph6 files. Covers:

- All graphs on 5–10 vertices
- Connected graphs on 6–10 vertices
- Cubic connected graphs on 8–20 vertices (even orders)
- Directed graphs on 1–6 vertices

**Dependencies:** `networkx`

### Example

```python
from itertools import islice

from pyg6data.lists import list_graphs, graph_generator

# Load all 112 connected graphs on 6 vertices into memory
graphs = list_graphs(6)
print(len(graphs))               # 112

# graph_generator is lazy: take only the first five connected 9-vertex graphs
for g in islice(graph_generator(9), 5):
    print(g.number_of_edges())   # graphs are produced one at a time

# Note: exhausting the generator (e.g. with sum(...)) visits all ~261 000 graphs

# Run a filter/evaluator census with an atomic, resumable checkpoint
from pyg6data.census import run_graph_census

records = run_graph_census(
    6,
    lambda g: g.number_of_edges() > 5,
    lambda g: g.number_of_edges(),
    checkpoint="census.json",
)
```

---

## pycombtop

Combinatorial topology tools for graphs and simplicial complexes:

- `SimplicialComplex` / `Simplex` — core data structures
- `clique_complex` — build the clique complex of a graph
- `homotopy_type_with_verdict` — determine the homotopy type of a graph's clique
    complex, with optional vertex/simplex limits that return explicit
    inconclusive results
- `fundamental_group` — compute the fundamental group (algorithm from [RS00], Rees & Soicher, *J. Symbolic Comp.* 29, 2000; see [`docs/references.rst`](docs/references.rst))
- `complete_s_collapse` — strong collapses
- `hom_graph` / `graph_homomorphisms` — Hom complexes

**Dependencies:** `networkx`, `pycliques`, `sympy`, `mogutda`

### Example

```python
import networkx as nx
from pycombtop import clique_complex, homotopy_type_with_verdict

g = nx.cycle_graph(5)
v = homotopy_type_with_verdict(g)
print(v.verdict)    # \(S^{1}\)  — the clique complex of C_5 is homotopy-equivalent to S^1
print(v.reason)     # the theorem that established the result
```

---

## pyhomrep

Computes **representations of groups on the homology** of simplicial complexes, with a focus on symmetric-group actions. Built on `sympy` for exact arithmetic.

- `character_p_homology` — character of a permutation on H_p
- `decompose_into_irreducibles` — decompose H_p as an S_n-module
- `MatrixRepresentation` / `regular_representation` — matrix representations
- `matching_graph` — construct the matching graph on which S_n acts

**Dependencies:** `networkx`, `numpy`, `pycliques`, `pycombtop`, `sympy`

### Example

```python
import networkx as nx
from sympy.combinatorics import Permutation
from pycombtop import clique_complex
from pyhomrep import character_p_homology
from pyhomrep.graphs import matching_graph

g = matching_graph(5)
sc = clique_complex(g)
# Character of the identity permutation on H_0 equals 1 (one connected component)
print(character_p_homology(sc, 0, Permutation()))   # 1
```