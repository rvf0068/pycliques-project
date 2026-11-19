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

## One Morse 1-circle killed by a primitive 2-dimensional attachment

This extends the preceding critical-0/1-pair rule to an augmented
Morse matching with **multiple** critical 0- and 1-cells, provided the
restored 1-skeleton has *exactly one independent circle*. In the exact
Morse CW model, write `c_j` for critical `j`-cells and `D_j` for
the integral cellular boundary, remembering that unaugmenting the
matched empty face adds one base 0-cell.

The new recognition check requires:
- No critical cells above dimension 3, `c_1 = c_0 + 1`, and
  `c_2 > 0`.
- `D_1` has full row rank `c_0` with all nonzero Smith invariants
  equal to one. Thus the 1-skeleton is a *connected graph with
  exactly one circle*.
- `D_2` has rank **one** with its only nonzero Smith factor equal
  to one. The 2-cells attach by powers of the circle generator with
  greatest common divisor one, so **pi_1 of the 2-skeleton is
  trivial**. This is a fundamental-group argument, *not* an inference
  from vanishing `H_1` for an arbitrary graph presentation.
- `D_3` has only unit nonzero Smith factors. All matrices have the
  expected shape, their consecutive products vanish, and the exact
  torsion/Betti data are consistent.

A simply connected 2-dimensional CW complex with free `H_2` is
homotopy equivalent to a wedge of 2-spheres, by Hurewicz and
Whitehead. Since `pi_2 = H_2`, the primitive higher degree vectors
can be reduced by unimodular cellular changes of basis. If `r_3`
is the rank of `D_3`, the certified type is

```
(c_2 - 1 - r_3) copies of S^2
(c_3 - r_3) copies of S^3
```

Negative multiplicities, nonunit invariants, more than one circle,
or any unsupported attachment pattern return an *inconclusive* result.

The 48-vertex, 163-edge reduced core of
`K^4(Susp(C_12))` is the regression example. Its critical counts
are `{0:5, 1:6, 2:17, 3:1}`; the Smith diagonals of
`D_1,D_2,D_3` are respectively
`(1,1,1,1,1)`, `(1)`, and `(1)`. Thus its clique complex is
`vee_15 S^2`, and suspending the independently certified large
factor reduction yields `vee_15 S^3` for the fourth iterate.

Tests independently enumerate its 387 augmented simplices, verify
the full reoriented matching Hasse DAG, and check simple connectivity
by spanning-tree triangle relations, in addition to checking the
exact integral Morse matrices. This rule is applied only when the
bounded symbolic integral Morse search already runs; it does not
introduce expensive graph reductions into the default classifier.

## A unit 0--1 Morse pair followed by consecutive-dimensional cells

An exact augmented Dong matching may leave one critical vertex, one
critical edge, and critical simplices in dimensions `d,d+1` with
`d >= 2`. The new symbolic integral Morse recognition rule also
handles this case.

The checker requires a **unit** integral differential
`d_1 = (+1)` or `d_1 = (-1)`, zero intermediate differentials,
and a `d_(d+1)` whose every nonzero Smith invariant is `1`.
There must be no other critical dimensions and all integral homology
torsion must vanish. These conditions are verified on the already
certified **acyclic** augmented matching; the check never infers a
homotopy type just from Betti numbers.

After restoring the vertex paired with the augmented empty face,
the two 0-cells and the critical 1-cell form a **contractible tree**.
Collapsing it leaves a pointed CW complex with cells only in dimensions
`d` and optionally `d+1`. Because `d >= 2`,
`pi_d(vee S^d) = H_d(vee S^d; Z)`; unit Smith invariants make
the nonzero attaching degrees cancellable. Hence the remaining
critical cells determine a wedge of spheres.

```python
from pycombtop import symbolic_integral_morse_certificate

proof = symbolic_integral_morse_certificate(core)
if proof is not None and proof.wedge_spheres is not None:
    print(proof.wedge_spheres)  # proven wedge multiplicities
```

For the fixed **70-vertex, 342-edge** core of the suspension factor
of `K^4(Susp(C_11))`, the augmented matching has
`{0:1,1:1,2:3,3:4}` critical cells and integral boundaries
`d_1=(1)`, `d_2=0`, `d_3=0`. After cancelling the 0--1 pair, the **certified**
core homotopy type is `(vee_3 S^2) vee (vee_4 S^3)`.
Suspending recovers
`(vee_3 S^3) vee (vee_4 S^4)` for the fourth clique iterate.

The verification tests independently enumerate all **1,215** augmented
faces of that core, check acyclicity of the explicit matching's
reoriented Hasse graph, and compare exact symbolic integral Morse
boundaries and the classifier verdict.

## Certified Hopf cup-square obstructions for nonconsecutive Morse cells

`hopf_wedge_certificate(G)` closes a specific gap left by ordinary integral
Morse boundaries: a connected flag complex with an augmented, acyclic
symbolic Dong matching whose Morse CW model has **one essential 2-cell**
and some **4-cells**, possibly with cancelling 0/1 and 3/4 pairs.
The verifier demands all of:

1. Exactly one critical 2-cell; at most one critical 0/1 pair, whose
   differential has degree one; no critical cells above degree 4
   or in other dimensions.
2. The integral 2- and 3-cellular boundaries vanish; the
   4-to-3 differential has **surjective unit Smith invariants**.
   Thus the 3-cells can be cancelled, yielding a CW model consisting
   of `S^2` with `k` 4-cells attached. The integral homology is
   torsion-free with `H_2=Z` and `H_4=Z^k`.
3. A maximal triangle whose indicator defines an integral 2-cocycle
   representing a **nonzero** class modulo 2. This is checked using
   exact bitset Gaussian elimination on the simplicial
   coboundary `C^1(F_2) -> C^2(F_2)`, not Betti numbers alone.

The triangle indicator's integral Alexander–Whitney cup square is
**identically zero as a cochain**: the initial and final 2-faces of an
ordered 4-simplex cannot both equal that same triangle. The indicator
is an *odd* multiple of the integral `H^2=Z` generator, so its square
vanishing in torsion-free `H^4` forces the generator's square to vanish.
That square records the Hopf invariant of each attaching map in
`pi_3(S^2)=Z`. Therefore **all** 4-cells attach trivially, and the
replayer certifies exactly `S^2 vee k*S^4`.

```python
from pycombtop import hopf_wedge_certificate, verify_hopf_wedge

proof = hopf_wedge_certificate(core)
if proof is not None:
    assert verify_hopf_wedge(core, proof)
    print(proof.spheres)  # exact homotopy type, e.g. {2: 1, 4: 6}
```

The usual `homotopy_type_with_verdict` now invokes this *only if* its
symbolic integral Morse matching is otherwise inconclusive and the
narrow 2/4-dimensional pattern applies. The optional
`reduced_hopf_wedge_certificate` first executes the existing
replayable edge/vertex reductions on a large graph, then verifies
the Hopf wedge on the small remaining core. None of the large-graph
search is enabled implicitly in the homotopy census.

The new regression uses the 61-vertex, 417-edge core obtained from
`K^4(Susp(C_10))`: exact critical cells
`{0:1,1:1,2:1,3:2,4:8}`, unit rank-two `d_4`, and a
maximal triangle `(47,50,59)` representing an odd cohomology
generator. The resulting type is `S^2 vee 6*S^4` for the core,
or `S^3 vee 6*S^5` after the independent suspension reduction.

The method deliberately returns **None** for general higher-dimensional
attachments (including nontrivial Hopf invariants), insufficient
cohomology witnesses, unsupported CW patterns, and resource limits.

References: Forman (1998), Hatcher *Algebraic Topology*, Section 4.B
(Hopf invariant and cup square), and the Alexander–Whitney cup product.

## Certified dense flag-edge reductions and vertex-star wedges

The opt-in API `reduce_flag_graph(G)` combines three independently
replayable homotopy-preserving moves:

- **Cone-link edge removal:** for an edge uv, a stored apex is adjacent
  to every other vertex in the common-neighbor graph, making its flag
  complex contractible.
- **Dominated-vertex fold:** a recorded adjacent vertex has a superset of
  the removed vertex's closed neighborhood.
- **Contractible-link edge removal:** once the graph has at most 80
  vertices, a zero-critical-cell augmented Dong matching certifies that
  an edge's *nonempty common-neighbor clique complex* is contractible.
  This extends cone-link removal to contractible links without an apex.

In a flag complex, deleting an edge with contractible link preserves
the homotopy type via the homotopy pushout
\(\Delta(G-e)\cup(e*\Delta(N(u)\cap N(v)))\).
The reduction certificate records **all** moves and the final graph.
`verify_flag_reduction(G, certificate)` replays local proofs on the
changing graph; it never trusts the search's cached adjacency bitsets.

For a small residual core, `vertex_star_wedge_certificate` computes
augmented Dong matching critical-cell counts for a deletion and its
link, and accepts the resulting CW models only when these patterns
prove exact wedges. In particular, a *connected* CW model with two
0-cells, a single 1-cell, and higher critical cells has a contractible
1-skeleton; with one 0-cell, circles, and cells only in dimension
at least 3, the higher attachments are null since higher homotopy
groups of a wedge of circles vanish.

When the deletion has a certified wedge whose lowest sphere dimension
is **strictly higher** than every link sphere dimension and is at
least 2, the link inclusion is nullhomotopic. Vertex-star cofiber
splitting then gives a verified wedge for the core. The wrapper
`reduced_star_wedge_certificate(G)` combines both steps and
`verify_reduced_star_wedge(G, proof)` checks the whole proof again.

```python
from pycombtop import reduced_star_wedge_certificate, verify_reduced_star_wedge

proof = reduced_star_wedge_certificate(G, max_vertices=1000)
if proof is not None:
    assert verify_reduced_star_wedge(G, proof)
    print(proof.spheres)  # exact clique-complex homotopy type
```

This search is **opt-in**, with caps on moves, Dong diagram nodes,
and allowable residual graph/link order; failure is inconclusive and
does not change the normal homotopy census classifier. The optional
`candidates` argument specifies vertex indices in the canonically
relabelled reduced core. In the regression for the 801-vertex factor
of `K^4(Susp(C_9))`, this strategy reduces to 54 vertices and
491 edges and certifies `S^2 vee 2 S^6`; suspending yields
`S^3 vee 2 S^7` for the original fourth clique iterate.

## Symbolic Dong matching via zero-suppressed decision diagrams

The public `symbolic_dong_certificate(G)` implements **the same vertex-ordered
augmented discrete Morse matching** as `dong_matching` [Dong2003]. Its
zero-suppressed decision diagram (ZDD) stores a family of simplices exactly,
without materializing individual simplices. At each pivot `v`, if the
remaining face family is `F_minus union (v * F_plus)`, it removes both copies
of `F_minus intersection F_plus`; the remaining branches are the two set
differences. Dong's sequential matching theorem proves acyclicity, while
the ZDD operations compute the resulting critical cells exactly.

The certificate records the critical-cell counts by dimension, including
dimension -1 for an unmatched empty face, the number of faces (including the
empty face), the number of ZDD nodes, and all critical-face bitmasks when
there are at most 256. The default ordering is numeric for integer-labeled
graphs and stable for other labels; specify `vertex_order` to replay a
particular matching. The search has conservative graph-order and ZDD-node
caps; an exceeded cap returns `None`, not a proof.

```python
from pycombtop import symbolic_dong_certificate, homotopy_type_with_verdict
import networkx as nx

g = nx.cycle_graph(5)
certificate = symbolic_dong_certificate(g)
assert certificate is not None and certificate.critical == {1: 1}
assert homotopy_type_with_verdict(g).is_exact
```

For graphs with **32 to 80 vertices**, `homotopy_type_with_verdict`
automatically tries this bounded symbolic proof **before** imposing the
ordinary `max_simplices` limit. If the empty face is paired and either no
critical cells remain or all surviving cells have one nonnegative dimension,
the corresponding wedge of spheres is exact. Mixed-dimensional critical
cells alone do *not* establish a wedge, and the ordinary classifier continues
with its other methods when symbolic matching is inconclusive.

For example, the fixed 56-vertex graph
`K^3(Susp(C_6))` has 657,325,163 faces including the empty face, but the
symbolic matching returns precisely eleven critical 7-cells, establishing
`Delta(K^3(Susp(C_6))) ~= wedge_11 S^7`. This case is covered by a
reproducible fixed-graph6 regression, which also checks
`homotopy_type_with_verdict(G, max_vertices=64, max_simplices=25000)`.

This is **an optimized implementation of Dong's matching, not a new
discrete Morse matching theorem**. Unlike the explicit matcher, it does not
materialize all face pairs for an independent DAG acyclicity check; its
acyclicity guarantee comes from the mathematically established sequential
rule. Tests compare the symbolic critical faces against both an independent
face-set implementation and `dong_matching` on randomly generated small
graphs.

## Integral Morse homology from symbolic Dong matchings

`symbolic_integral_morse_certificate(G)` extends the ZDD-backed Dong
matching to **mixed-dimensional critical cells** without expanding its face
poset. It retains the decreasing family of remaining faces at each pivot;
an exact face-membership lookup identifies which pivot matched a face.
Signed gradient-path recursion then computes integral Morse boundary matrices.
Smith normal form gives **reduced integral Betti numbers and torsion**.

The certificate reports `critical`, `boundaries` (rows indexed by lower
critical masks, columns by upper critical masks), `betti`, `torsion`,
`wedge_spheres` (possibly `None`), and resource counts. ZDD-node,
critical-cell, and gradient-state limits fail closed: `None` means no
certificate, not a vanishing invariant.

One positive-dimensional critical degree certifies a wedge immediately.
For **exactly two consecutive critical dimensions** `d,d+1` with
`d>=2`, the attaching maps are classified by their degree vectors in
`pi_d(wedge S^d)`. If all nonzero Smith invariants of the integral
Morse boundary are **units**, elementary cellular cancellations give
a wedge of spheres. In every other mixed-dimensional situation the
boundary still computes integral homology but does **not** determine
the homotopy type. A matrix with zero boundary is not, in general,
a proof of nullhomotopic higher attaching maps.

```python
from pycombtop import symbolic_integral_morse_certificate

certificate = symbolic_integral_morse_certificate(G)
if certificate is not None:
    print(certificate.betti, certificate.torsion)
    if certificate.wedge_spheres is not None:
        print("Certified wedge:", certificate.wedge_spheres)
```

For the 56-vertex graph `K^3(Susp(C_7))`, comprising **546,132,471**
augmented faces, the integral matching has critical cells in dimensions
3 (one) and 4 (two), with `boundaries[4] == ((0, 0),)`.
It certifies `S^3 wedge S^4 wedge S^4`; the standard
`homotopy_type_with_verdict` automatically uses this method when
a 32–80 vertex symbolic matching has mixed critical dimensions.
Tests use a fixed graph6 encoding and also check a nonzero Morse
differential, ensuring that integral boundary computations are not
silently replaced with Betti-based guesses.

## Weighted conicity and certified connectivity

`verify_weighted_conicity(G, weights)` uses **integer arithmetic only**.
Let `W` be the positive sum of nonnegative vertex weights and `M`
the maximum weight of `N_complement[v]`, the closed neighborhood
of `v` in the complement of `G`. If `r*M < W`, no `r` closed
neighborhoods can cover all vertices; hence every set of at most
`r` vertices has an external common neighbor in `G`. The clique
complex is `r`-conic, and for `r>=2`, Barmak's conicity
theorem makes it `(floor(r/2)-1)`-connected.

`find_weighted_conicity_certificate` optionally uses SciPy's HiGHS linear
program to propose optimized weights. The candidate is rounded to
integers and **reverified exactly** before return, and a uniform weight
certificate is retained when optimization is inconclusive. These are
connectivity results, **not certificates of an exact homotopy type**.

```python
from pycombtop import find_weighted_conicity_certificate

proof = find_weighted_conicity_certificate(G, time_limit=10)
if proof is not None:
    print(proof.conicity, proof.connectivity)
```

For the 679-vertex suspension factor `H` in
`K^4(Susp(C_7)) = Susp(H)`, a verified integer-weight witness gives
`W=10,000,010`, `M=1,665,331`, thus 6-conicity of `H`
and 2-connectivity of its clique complex. Suspension implies
3-connectivity of `Delta(K^4(Susp(C_7)))`.

References: Forman, *Morse Theory for Cell Complexes* (1998);
Dong, *Topology of Bounded-Degree Graph Complexes* (2003);
Hatcher, *Algebraic Topology*, Section 4.2;
Barmak, *Connectivity of Ample, Conic, and Random Simplicial
Complexes*, IMRN (2023), and Farber, *Large Simplicial Complexes*
(2024), connectivity appendix.

## Phase-2 budget recovery and reproducible Dong search

For graph inputs, `homotopy_type_with_verdict` retains the strict
`max_vertices` check on the **original** graph. The `max_simplices` limit
guards general flag-complex methods: cheap exact cone and join certificates
run first. If the original face-count bound is exceeded, the classifier now
alternates graph vertex and edge s-collapses until stable, checks cheap exact
certificates on that smaller graph, and reruns the face-count guard on the
reduced core. An irreducible core that still exceeds the bound remains
`LIMIT_EXCEEDED`, not an inferred type. S-collapses preserve the
simple-homotopy type of the clique complex.

`homotopy_type_with_verdict(graph, dong_seed=0)` and
`homotopy_type_sc_with_verdict(complex, dong_seed=0)` use a local seeded
generator for Dong's six shuffled vertex orderings; the initial ordering
and each link's simplex ordering are stable for ordinary labeled vertices,
including tuple and frozenset clique labels. No global random state is
consumed. The explicit seed affects Dong attempts in the current call;
recursively classified factors use their default seed. All returned exact
Dong types still require independent acyclicity verification. A seed
controls search order, **not** mathematical correctness, and it does not
remove the need to save and check mathematical certificates for new results.

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

## Multi-facet puncture certificates for wedges of spheres

The function `facet_puncture_wedge_certificate(complex_)` implements a
bounded, exact topological recognition strategy for arbitrary finite
simplicial complexes (not only flag complexes). It first performs
genuine elementary collapses to a residual complex (R), then tries
removing several **maximal facets of one dimension**. If the remaining
complex collapses to a single vertex, it returns a certificate.
`verify_facet_puncture_wedge` independently replays every free
face--coface pair and checks the facet deletions.

**Theorem.** Suppose (X\searrow R), and (d\ge1) and distinct
maximal (d)-simplices \(\sigma_1,\ldots,\sigma_m\) of (R)
have the property that
\(A=R\setminus\{\sigma_1,\ldots,\sigma_m\}\) is a subcomplex
collapsing to a vertex. Then
\[
X\simeq R\simeq R/A\simeq \bigvee^m S^d.
\]
Here only the maximal simplices are removed: **all proper faces
remain**. Since (A) is contractible and (A\hookrightarrow R) is
a CW-cofibration, the quotient map is a homotopy equivalence. The
quotient is exactly (m) (d)-cells with their boundaries collapsed
to one point, giving the stated wedge.

The necessary number (m) of removed facets equals the highest
mod-2 Betti number (\widetilde\beta_d(R;\mathbb F_2)). The
implementation calculates this rank exactly to avoid searching all
possible subset sizes. Failure to find a certificate within
`max_candidates`, `max_faces` or `max_removed_facets`
is **inconclusive**, not a non-wedge proof.

The following reproduces the common 16-vertex graph encountered in
the two unresolved nine-vertex sphere calculations of
`pycliques-census`:

```python
import networkx as nx
from pycombtop import (
    clique_complex, facet_puncture_wedge_certificate,
    verify_facet_puncture_wedge,
)

prism_complement = nx.complement(nx.circular_ladder_graph(8))
complex_ = clique_complex(prism_complement)
certificate = facet_puncture_wedge_certificate(complex_)
assert certificate is not None
assert certificate.spheres == {3: 3}
assert verify_facet_puncture_wedge(complex_, certificate)
```

The algorithm detects a wedge of three 3-spheres from 1,154
nonempty simplices by retaining complete, replayable certificates.
A separate s-collapse preprocessing stage, already available as
`complete_s_collapse_with_certificate`, can reduce a larger flag
complex to a tractable core before this general simplicial method
is applied.

## Recursive vertex stars and integral Morse boundaries

`recursive_wedge_certificate(G)` takes a finite simple graph with arbitrary
hashable vertex labels. It searches for a proof by dominated-vertex folds,
joins, and vertex-star decompositions. A split requires a proof that the
link inclusion into the deletion is nullhomotopic. The optional
`max_vertices` and `max_states` bound the search.

`integral_morse_certificate(facets)` takes the facets of any finite
simplicial complex as a set of positive integer bitmasks. Bit position \(i\) represents
vertex \(i\). It replays free pairs and critical faces and, when the positive
critical dimensions are consecutive and the lower one is at least two,
checks the signed integral Morse boundary. The optional `max_faces` bounds
the number of nonempty faces. For a graph with connected bipartite
complement, `bipartite_shore_facets(G)` can supply a smaller complex whose
suspension has the desired homotopy type.

Both searches return a certificate with a `spheres` dictionary on success
and `None` when the search is inconclusive. The corresponding verifiers
can replay a certificate independently of the search.

```python
import networkx as nx
from pycombtop import recursive_wedge_certificate, verify_recursive_wedge

graph = nx.complete_bipartite_graph(2, 3)
proof = recursive_wedge_certificate(graph)
assert proof is not None and proof.spheres == {1: 2}
assert verify_recursive_wedge(graph, proof)
```
