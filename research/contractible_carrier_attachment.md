# Contractible carriers for vertex-star attachments

## Setup and conventions

Let \(H\) be a finite simple graph. A **complete** is a nonempty
pairwise adjacent set, and a **clique** is a maximal complete. Write
\(\Delta(H)\) for the flag complex of \(H\).

Fix a vertex \(v\) and let
\[
 B=\Delta(H-v),\qquad L=\Delta(H[N_H(v)]).
\]
The full star of \(v\) is the simplicial cone \(v*L\); therefore
\[
 \Delta(H)=B\cup_L(v*L).
\]
This is a model for the **homotopy cofiber** of the inclusion
\(j:L\hookrightarrow B\) (simplicial inclusions are cofibrations).

**Null-attachment theorem.** Suppose a graph \(D'\) is obtained from
\(H-v\) through ordinary dominated-vertex folds, and let \(r:B\to
\Delta(D')\) be their composite simplicial retraction. Suppose there
exists an induced subgraph \(A\subseteq D'\) such that (a) the image of
every vertex of \(N_H(v)\) under \(r\) lies in \(A\) and (b)
\(\Delta(A)\) is contractible. Then
\[
 \Delta(H)\simeq B\vee\Sigma_{\mathrm u}L.
\]
Here \(\Sigma_{\mathrm u}L\) is the *unreduced* suspension, including
the extra circles arising when \(L\) is disconnected.

### Proof in full

1. Each ordinary graph fold deletes a vertex \(x\) dominated by a
   different adjacent vertex \(y\):
   \(N_D[x]\subseteq N_D[y]\). Sending \(x\) to \(y\) and fixing all
   other vertices defines a simplicial map
   \(\Delta(D)\to\Delta(D-x)\). The inclusion is a right inverse.
   The composites are contiguous to the identity: for every simplex
   \(\sigma\), \(\sigma\cup r(\sigma)\) is still a simplex, by the
   closed-neighborhood inclusion. Consequently **each ordinary fold is
   a homotopy equivalence**, and so is the composite \(r\).

2. Because the image of the link lies in \(A\), the simplicial
   composite \(r\circ j:L\to\Delta(D')\) factors through
   \(\Delta(A)\). Condition (b) makes this composite null-homotopic.
   Since \(r\) is a homotopy equivalence, composition with a homotopy
   inverse shows \(j\) itself is null-homotopic.

3. The homotopy cofiber of a null-homotopic map
   \(j:L\to B\) is homotopy equivalent to
   \(B\vee\Sigma_{\mathrm u}L\). This gives the stated equivalence.

Notice that *no dimension inequality* is used. In contrast, the
earlier criterion proved null-homotopy using the connectivity of the
target sphere wedge, and required every source-sphere dimension to
be strictly smaller than the least target-sphere dimension.

### How the carrier is certified

The algorithm does **not** use homology to assert contractibility.
Instead, it records a sequence of vertex **\(s\)-dismantlings** of
\(A\) down to a single vertex. Each removed vertex \(w\) comes with
a complete dominated-vertex sequence reducing its **open
neighborhood** \(N_A(w)\) to a singleton.

An ordinary domination sequence of a graph \(N_A(w)\) proves that
\(\Delta(N_A(w))\) is contractible. Attaching the cone
\(w*\Delta(N_A(w))\) to \(\Delta(A-w)\) along this contractible link
does not change homotopy type. Thus each recorded \(s\)-dismantling
is a homotopy equivalence; a sequence ending in one vertex certifies
\(\Delta(A)\simeq\{\ast\}\).

The independently replayed certificate checks, in order:

- every ordinary deletion-side domination \(N[x]\subseteq N[y]\);
- the full image set of the original link vertices in the folded graph;
- that the carrier vertices contain this full image;
- every local dominated-vertex move in every \(s\)-dismantling;
- that the carrier has **exactly one surviving vertex**;
- the exact wedge-of-spheres type of the folded deletion, by the
  existing structural recognizer.

If any check fails, the verifier returns **false**, and the
homotopy classifier does not emit an exact verdict.

## A 15-vertex example: a carrier with no cone point

Start with a five-cycle on vertices \(0,1,2,3,4\). Attach the
path \(0-5-6-7\). Add six new vertices \(8,\ldots,13\), inducing
a six-cycle, and prescribe target vertices
\[
 (f(8),f(9),f(10),f(11),f(12),f(13))
       =(0,5,6,7,6,5).
\]
For each new vertex \(t\), join \(t\) to \(f(t)\) and to all
neighbors of \(f(t)\) in the original eight-vertex graph.
Leave the induced edges on \(8,\ldots,13\) exactly those of \(C_6\).

Let \(D\) denote this 14-vertex graph, and form \(H\) by adding
vertex \(14\), adjacent precisely to \(8,\ldots,13\).
Then \(L=\Delta(C_6)\simeq S^1\).

The following six **ordinary** folds are valid in this order:
\[
 8\mapsto0,\quad9\mapsto5,\quad10\mapsto6,\quad
 11\mapsto7,\quad12\mapsto6,\quad13\mapsto5.
\]
They leave precisely the original \(C_5\) with the pendant path
\(0-5-6-7\), so the resulting deletion complex has type \(S^1\).
They map the link onto the four vertices
\(\{0,5,6,7\}\), inducing the path \(P_4\).

The image **has no universal vertex**: the cone-image criterion
therefore does not apply to this folded target. Nevertheless
\(\Delta(P_4)\) is contractible (remove its successive end vertices,
each with an explicitly dismantlable open neighborhood). Hence the
new contractible-carrier criterion gives the exact homotopy type
\[
 \boxed{\Delta(H)\simeq S^1\vee S^2.}
\]
Both \(L\) and \(B\) have type \(S^1\); in particular, the older
sphere-dimension inequality is **not** satisfied. The test fixture
uses the same six domination moves and checks all their certificates.

The previous example of order-10 census index \(10223998\)
remains a cone-image certificate and is not reclassified.

## Computational limits

The classifier's **single shared deletion-fold search** performs
at most 18 folds and 1024 domination comparisons, on a deletion
with at most 24 vertices. The extra carrier search is performed only
after the cone test fails; it tries at most 36 carriers, each with at
most 12 vertices, at most two vertices outside the image, and at most
64 local \(s\)-move checks per carrier. The pre-existing outer
algorithm considers at most two deletion candidates.

These are search limits, not hypotheses of the theorem. Exceeding a
limit returns no certificate, without making a false homotopy claim.

## Background references

- A. Hatcher, *Algebraic Topology*, Cambridge University Press,
  2002, Chapter 0 (mapping cones, homotopy equivalences, and
  homotopy-cofiber constructions).
- D. N. Kozlov, *Combinatorial Algebraic Topology*, Springer,
  2008, chapters on collapses and simple-homotopy.
- R. Boulet, É. Fieux, and B. Jouve, *Simplicial simple-homotopy of
  flag complexes in terms of graphs*, European Journal of
  Combinatorics **31** (2010), for graph \(s\)-dismantling
  and simple-homotopy of flag complexes.
