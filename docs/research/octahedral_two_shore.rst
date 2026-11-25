Two-shore exact retractions onto induced octahedra
==================================================

Let X be a finite simple graph and T an induced octahedron
O_d = complement(d K_2), with antipodal pairs (a_i,b_i).
Graph morphisms are weak: an edge may map to an edge or collapse
to a single vertex.

The public function find_octahedral_retraction_shores searches
for full ordinary retractions onto a specified induced octahedron.
It does not use a saved projection or search for a target embedding.

Theorem: complete forced two-shore branching
--------------------------------------------

For every vertex x outside the target, put

.. math::

   A(x) &= \{i : xb_i\notin E(X)\},\\
   B(x) &= \{i : xa_i\notin E(X)\}.

Then x may map to a_i exactly when i is in A(x), and to b_i
exactly when i is in B(x), as far as its edges to the fixed
target are concerned. Each target vertex has exactly one
distinct target nonneighbor: its antipode.

Suppose a set Y of outside vertices has been assigned
r(y)=b_{\beta(y)}, where \beta(y)\in B(y). Define

.. math::

   A_Y(x)=A(x)\setminus
           \{\beta(y) : y\in Y,\ xy\in E(X)\},\qquad x\notin Y.

**Theorem.** The following algorithm is sound and complete
for all retractions X -> T fixing T pointwise.

1. Initially Y is empty.
2. If an outside unassigned x has A_Y(x) empty, branch over
   every i in B(x), assign r(x)=b_i and update A_Y for its
   adjacent unassigned vertices. If B(x) is empty, reject
   that branch.
3. If no such x remains, assign each unassigned outside x
   to any a_i with i in A_Y(x). This completes a retraction.

**Proof.** Within either shore, all target vertices are
adjacent or equal. The only forbidden image of an edge
between different shores is an antipodal pair a_i,b_i.
The initial domains A(x) and B(x) enforce compatibility
with the fixed target. Assigning y to b_j excludes a_j
at every outside neighbor of y, precisely the update in
A_Y. If A_Y(x) is empty, any extending retraction must
send x to some b_i, i in B(x), and the algorithm tries
every such choice. If no empty domain remains, every
unassigned vertex may be sent to an allowed a_i:
a-a images never conflict, and all a-b conflicts
with previously assigned b-images have been removed.
Induction along the finite search tree proves that every
successful return is a retraction and every possible
retraction is represented by at least one branch. QED.

The proof is independent of how the antipodal pairs are
oriented. Trying a promising orientation first is strictly
a search-order optimization; an EXHAUSTED search for even
one fixed orientation excludes retractions to that target.

API and exact statuses
----------------------

.. code-block:: python

    import pycliques as pc

    result = pc.find_octahedral_retraction_shores(
        graph,
        pairs,
        max_states=20_000,
        orientation=None,
    )
    if result.status is pc.RetractionSearchStatus.FOUND:
        target = graph.subgraph(v for a, b in pairs for v in (a, b))
        assert pc.verify_retraction_witness(
            graph, target, (result.retraction, result.inclusion)
        )

The integer orientation is a bitmask that swaps the first
and second endpoints of each specified antipodal pair.
When orientation is None, a deterministic heuristic picks
one orientation with few initially forced b-shore vertices.
The sampling budget is max_orientation_samples (default
65,536). It affects speed only, not correctness.

FOUND returns a complete map, independently checked on
all graph edges. EXHAUSTED proves there is **no ordinary
retraction onto the specified induced octahedron**.
LIMIT_REACHED is inconclusive and must never be promoted
as a negative theorem. The initial search state and each
attempted b-shore assignment count against max_states.

The implementation uses iterative DFS with rollback trails,
so it is not limited by the Python recursion depth.

Research calibration
--------------------

An independent prototype was tested against two complete,
previously certified retraction targets from
https://github.com/rvf0068/pycliques-census/pull/80.

===================  =================  ==========================
Nine-vertex McKay     Certified target   Two-shore search states
===================  =================  ==========================
251759                P4 -> O15           3
254369                P4 -> O13           33
===================  =================  ==========================

Both full returned maps passed independent edge preservation
on their 403- and 394-vertex hosts, respectively. In the
best orientations, a single-shore target cover left two
and one exceptional outside vertices. The new search
handles these through forced b-shore assignments.

These are positive replay results, not new classifications
of the unresolved census cases. The existing target-first
MRV and unseeded coaffine solvers are unchanged.
