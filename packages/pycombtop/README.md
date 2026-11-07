# Certified homotopy reductions

`homotopy_type_with_verdict(G)` analyzes the clique complex
`Delta(G)`. A result with `is_exact=True` is an asserted homotopy type;
a result with `is_exact=False` carries Betti numbers only and must not
be interpreted as a wedge-of-spheres certificate.

The following descriptions explain the *reason* recorded in each exact
verdict. They are mathematical arguments, not references to privately
generated reports.

- **Universal-vertex cone.** A vertex adjacent to all other vertices
  belongs to every maximal complete; its clique complex is a cone.
- **Graph join.** If the complement disconnects, the graph is a join
  of the induced complementary factors; their clique complexes form a
  simplicial join. We combine *certified* wedge types with the
  join-suspension formula, not Betti numbers alone.
- **Disjoint-complete open neighborhood.** The code checks that
  `N_G(v)` has at least two connected components and *each* induces a
  complete graph. It also explicitly checks that `G-v` is connected.
  The vertex star is a cone on the disjoint union of these contractible
  simplices, and its attachment contributes one circle per component
  beyond the first. Two isolated neighbors are included, as are three
  or more complete components.
- **Contractible-link components (arbitrary simplicial complexes).**
  This separately checks that the link has at least two connected
  components, that every component is a *certifiably contractible*
  subcomplex, and that the deletion is connected. Cone attachment
  then contributes `r-1` circles. This method does not require a flag
  complex.
- **Triangle-free intercomponent edges.** The algorithm verifies that
  removing selected edges leaves *exactly two connected induced parts*,
  that every selected edge crosses between them, and that none lies in
  a triangle. Attaching `m` one-cells between the two connected pieces
  contributes `m-1` circles.
- **Cutvertex wedge.** Decomposition at a genuine articulation vertex
  produces subcomplexes with precisely that vertex in common; their
  homotopy types form a wedge.
- **Dong discrete Morse matching.** The vertex-ordered matching is
  checked independently: every pair is a codimension-one face/coface
  pair, each face is used at most once, and the oriented face poset
  is acyclic. If all *remaining* critical cells have one dimension,
  Dong's application of Forman's discrete Morse theory gives a wedge
  of spheres. We retain the attribution to Dong and the 2003 publication
  year.
- **Vertex decomposability.** A shedding-vertex recursion checks the
  nonpure vertex-decomposable condition. Björner--Wachs then supplies
  shellability and a wedge-of-spheres homotopy type. Only in this
  certified case do reduced Betti numbers determine sphere counts.
- **Elementary collapse.** `collapse` removes only an actual free
  face paired with a unique codimension-one coface; the other faces
  survive. This is a simple-homotopy preserving operation, unlike
  arbitrary deletion of a face contained in one facet.

For the related clique-graph theorem, call:

```python
import networkx as nx
from pycombtop import homotopy_type_clique_graph_via_independent_neighborhood

G = nx.cycle_graph(6)
result = homotopy_type_clique_graph_via_independent_neighborhood(G, 0)
assert result.is_exact
assert result.wedge.spheres == {1: 1}
```

This function explicitly checks that `v` has at least two **pairwise
nonadjacent neighbors** and that `G-v` is connected of order at least
two. It determines the homotopy type of `Delta(K(G))` from
`Delta(K(G-v))`: the newly introduced maximal edge-completes form a
simplex in `Delta(K(G))`. Contracting that simplex reduces their
attachments to a cone on a disjoint union of simplices, so the
difference is a wedge of `degree(v)-1` circles. An invalid hypothesis
raises `ValueError`; an inexact recursive classification stays inexact.
The function avoids constructing `K(G)` itself, but does compute
`K(G-v)`.

## Literature used by the implementations

- Xun Dong, *Topology of bounded-degree graph complexes*,
  *Journal of Algebra* **262** (2003), 287--312,
  [doi:10.1016/S0021-8693(03)00024-3](https://doi.org/10.1016/S0021-8693(03)00024-3).
- Robin Forman, *Morse theory for cell complexes*,
  *Advances in Mathematics* **134** (1998), 90--145,
  [doi:10.1006/aima.1997.1650](https://doi.org/10.1006/aima.1997.1650).
- Anders Björner and Michelle L. Wachs, *Shellable nonpure complexes
  and posets. I*, *Transactions of the AMS* **348** (1996), 1299--1327,
  [author's bibliography](https://www.math.miami.edu/~wachs/eprints.html).
- Romain Boulet, Étienne Fieux and Bertrand Jouve,
  *Simplicial simple-homotopy of flag complexes in terms of graphs*,
  *European Journal of Combinatorics* **31** (2010), 161--176,
  [doi:10.1016/j.ejc.2009.05.003](https://doi.org/10.1016/j.ejc.2009.05.003).
- M. E. Frías-Armenta, F. Larrión, V. Neumann-Lara and M. A. Pizaña,
  *Edge contraction and edge removal on iterated clique graphs*,
  *Discrete Applied Mathematics* **161** (2013), 1427--1439,
  [doi:10.1016/j.dam.2013.02.003](https://doi.org/10.1016/j.dam.2013.02.003).
  Section 2 introduces local cutpoints; the independent-neighborhood
  homotopy reduction is proved directly in the code documentation.
- J. H. C. Whitehead, *Simplicial spaces, nuclei and m-groups*,
  *Proceedings of the London Mathematical Society* (2) **45**
  (1939), 243--327.

# Indeterminate graph files

Compute the homotopy type of each graph's clique complex and write a tab-separated
report alongside the input file:

```bash
uv run homotopy-indeterminate indeterminate_order_9.txt
```

The default output is `indeterminate_order_9_homotopy.txt`. Choose another path
with `--output`; use `--max-vertices` or `--max-simplices` to limit expensive
computations. The report preserves each graph's index, order, and graph6 string.
Its status is `exact`, `betti-only`, or `limited`, so inconclusive results are
not presented as proven homotopy types.

# Homotopy and collapse terminology

The greedy `collapse` routine does not preserve a subcomplex, and failure to
find another free face is not a proof of non-collapsibility. For a checked
relative certificate, use `relative_collapse` and independently validate its
elementary pairs with `verify_relative_collapse`. A found elementary collapse
is stronger than homotopy equivalence: it gives simple-homotopy equivalence,
whereas homotopy equivalence alone need not arise from such a sequence.
Simple-homotopy equivalence can also use expansions, so it does not itself
provide a collapse from this source to this fixed target. An inconclusive
search only reports the limits of the chosen algorithm.

`relative_betti_numbers` computes ordinary relative homology over :math:`Q`
with exact ranks by default; rational ranks do not detect integral torsion.
Vanishing relative Betti numbers are necessary for a collapse, but are not a
collapse certificate.

The distinction matters in clique-graph research. See Islas-Gómez,
“Propiedades homotópicas del operador de clanes en gráficas,”
[Pädi 10 (2022), DOI 10.29057/icbi.v10iEspecial.8511](https://doi.org/10.29057/icbi.v10iEspecial.8511),
and Larrión, Neumann-Lara, and Pizaña, “On the homotopy type of the clique
graph,” *J. Braz. Comp. Soc.* 7 (2001), 69–73. The octahedron is an important
exception to unqualified Whitney-surface homotopy claims; finite computations
are evidence only for the graph orders and budgets actually completed.

The bounded research census is available as
`uv run python research/homotopy_census.py 7` (or order `8`). Its JSON records
the graph6 value and NetworkX version, candidate filters, defect, a verified
relative-collapse certificate when found, and exact rational relative Betti
numbers when within the simplex budget. Budget-limited searches are explicitly
reported as `INCONCLUSIVE`, never as non-collapsible.

Order-7 and order-8 census JSON outputs are generated locally and are not
tracked in the repository. Reproduce the connected, labelled graph datasets
(considered up to isomorphism by the bundled graph6 data) with:

```sh
uv run python research/homotopy_census.py 7 --max-states 250 --max-simplices 500 --max-clique-graph-edges 100000
uv run python research/homotopy_census.py 8 --max-states 250 --max-simplices 500 --max-clique-graph-edges 100000
```

These process 853 and 11,117 connected graphs, respectively. The graph6 data
identifies each isomorphism class; `connected=True` is the census assumption.
Run `make research-test` for fast regressions including the order-7 candidate
count. The order-8 candidate count is a slower regression available through
`make research-test-slow`; CI runs it on schedule and manual dispatch.

For larger runs, stream JSONL records and resume safely with unchanged
parameters:

```sh
uv run python research/homotopy_census.py 8 --jsonl --output census-order8.jsonl \
  --max-states 250 --max-simplices 500 --max-clique-graph-edges 100000
uv run python research/homotopy_census.py 8 --jsonl --resume \
  --output census-order8.jsonl --max-states 250 --max-simplices 500 \
  --max-clique-graph-edges 100000
```

The JSONL header records Python, NetworkX, and package versions plus the order
and budgets. A separate `.summary.json` file is written when the dataset
finishes; resume rejects mismatched metadata and duplicate record indices.
The summary distinguishes a fully enumerated dataset from candidate-filter
results that were inconclusive for individual graphs.



## Bounded facet-greedy discrete Morse certification

The homotopy classifier now has a separate, exact facet-based matching
strategy in pycombtop.morse_certificates, complementing Dong's
vertex-ordered matching. It maintains maximal faces, repeatedly deletes
a free face--coface pair when available, and otherwise removes a maximal
simplex as a critical cell. The resulting matching is checked against
the full nonempty face-poset Hasse diagram for acyclicity.

A matching with one critical vertex and critical positive cells in
a single dimension immediately certifies a wedge of spheres. More
generally, with critical dimensions 0, m, m+1 for m>=2, if the exact
rational top Betti number equals the number of critical (m+1)-cells,
the cellular differential vanishes. The Hurewicz isomorphism for the
wedge of m-spheres then proves that every (m+1)-cell attaching map
is nullhomotopic. This is stronger than a Betti-only inference.

The method returns None if it cannot establish an exact certificate.
Its search is bounded by 12,000 distinct faces, 1,500 simultaneous
maximal faces, 12,000 reduction steps per attempt, three fixed seeds,
and a bounded exact top-dimensional boundary matrix. In the main
homotopy classifier, it precedes expensive recursion. High-dimensional
complexes with at least 20 facets are tested before Dong's matching;
otherwise it is tried later in the strategy cascade.

A direct certificate query (useful when an older classifier timed out):

~~~python
from pycliques import clique_graph
from pycombtop import graph_morse_certificate
import networkx as nx

g = nx.from_graph6_bytes(b"H?o~f~}")
certificate = graph_morse_certificate(clique_graph(g))
assert certificate is not None
assert certificate.spheres == {2: 1, 3: 2}
~~~

Six historical order-9 clique-graph timeout witnesses are covered by
regression tests. Their original completed census records should not
be modified in place; the separate pycliques-homotopy-census follow-up
script preserves the old runner and checksum provenance.

References: R. Forman, *Morse theory for cell complexes*, Adv. Math.
**134** (1998), 90--145; A. Hatcher, *Algebraic Topology* (2002),
Section 4.2 (Hurewicz theorem).
