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

**With uv** (recommended — handles the workspace automatically):

```bash
uv sync                        # install everything
uv add pycliques               # or add a single package to your project
```

**With pip** (install individual packages from source):

```bash
pip install ./packages/pycliques
pip install ./packages/pyg6data
pip install ./packages/pycombtop
pip install ./packages/pyhomrep
```

---

## pycliques

Tools for studying the **clique graph operator** K(G). Given a graph G, the clique graph K(G) has the maximal cliques of G as vertices, with two cliques adjacent when they share a vertex. The package provides:

- `clique_graph` — compute K(G) with an optional clique-count bound
- `is_clique_helly` / `is_hereditary_clique_helly` — Helly property tests
- `completely_pared_graph` / `find_dominated_vertex` — graph dismantling
- `recognize_clockwork` / `is_clique_divergent_clockwork` — clockwork-graph recognition
- `retracts` / `special_octahedra_dimension` — retraction tests
- CLI `small-behavior` — classifies all connected graphs of a given order

**Dependencies:** `networkx`, `grandiso`, `rich`  
**Optional:** `pyg6data` (needed only for the `small-behavior` CLI — install with `pip install "./packages/pycliques[data]"`)

### Example

```python
import networkx as nx
from pycliques.cliques import clique_graph
from pycliques.helly import is_clique_helly

g = nx.octahedral_graph()
print(is_clique_helly(g))        # False — not clique-Helly
kg = clique_graph(g)
print(kg.order())                # 8  (one vertex per triangle)
print(is_clique_helly(kg))       # False
```

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
from pyg6data.lists import list_graphs, graph_generator

# Load all 112 connected graphs on 6 vertices into memory
graphs = list_graphs(6)
print(len(graphs))               # 112

# Stream connected graphs on 9 vertices one by one
for g in graph_generator(9):
    if g.order() == 9:
        pass  # process without loading all ~261 000 graphs at once
```

---

## pycombtop

Combinatorial topology tools for graphs and simplicial complexes:

- `SimplicialComplex` / `Simplex` — core data structures
- `clique_complex` — build the clique complex of a graph
- `homotopy_type_with_verdict` — determine the homotopy type of a graph's clique complex
- `fundamental_group` — compute the fundamental group (algorithm from Rees & Soicher, *J. Symbolic Comp.* 29, 2000)
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