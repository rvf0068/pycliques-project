pycliques
=========

.. automodule:: pycliques
   :no-members:
   :show-inheritance:

Cliques
-------

.. automodule:: pycliques.cliques
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Clockwork
---------

.. automodule:: pycliques.clockwork
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Clockwork Target Libraries
--------------------------

A complementary approach to induced-subgraph search constructs a finite
library of theorem-certified divergent clockworks, then searches for
ordinary retractions onto these fixed targets. A positive verified
retraction certifies divergence; a negative result is restricted to the
targets actually checked. Neither construction nor recognition is a
complete enumeration of all divergent clockworks.

.. code-block:: python

   from pycliques.clockwork_library import generate_divergent_clockwork_library
   from pycliques.retractions import retracts, verify_retraction_witness

   library = generate_divergent_clockwork_library(
       [(1, 1, 1), (2, 1, 1)], crown_size=2
   )
   # G is a graph to classify.
   for target in library.targets:
       H = target.graph
       witness = retracts(G, H)
       if isinstance(witness, tuple):
           assert verify_retraction_witness(G, H, witness)
           break

The original order-nine census pilot (November 2026,
https://github.com/rvf0068/pycliques-census/pull/2) found 13 nonisomorphic
targets of orders nine and ten with these parameters. All eight tested
second pared clique iterates had no retraction onto the generated library.
The reported per-case runtime was 0.8--1.6 seconds; the earlier bounded
induced-subgraph pilot had seven timeouts and one candidate-limit result.
These are historical observations, not a negative result about other
clockworks or later iterates. The same PR reported exact homotopy agreement
between the original and first clique graph for all 54 historical cases
(46 spheres of dimension two and eight wedges of a circle and a
two-sphere); that pilot is superseded by the full homotopy census.

.. automodule:: pycliques.clockwork_library
   :members:
   :show-inheritance:
   :member-order: bysource

Clockwork Search
----------------

The bounded search functions below inspect induced vertex subsets in
increasing order. ``ClockworkSearchStatus.EXHAUSTED`` means the requested
candidate class was fully checked, while ``LIMIT_REACHED`` reports an
interrupted search; neither a limit nor failure to find a candidate proves
that no larger or unsearched clockwork exists.

An induced divergent clockwork is structural evidence, not a divergence
certificate for the containing graph. A verified graph retraction onto a
divergent clockwork does certify divergence, because the retraction theorem
transfers clique divergence from the retract to the source graph.

For example, ``nx.from_graph6_bytes(b"IEjdnpy^O")`` has a first clique graph
with a verified induced divergent clockwork on nine vertices. That
observation alone does not classify the first clique graph as divergent; the
bounded retraction search must separately find and verify a retraction.

Retraction search bounds candidate subsets, target attempts, backtracking
states, and assignments independently. Recognition limits are passed through
to both candidate recognition and dominated-vertex reduction. The classifier's
direct clockwork fallback is disabled by default; enable it explicitly with
``clockwork_fallback=True`` when using
``classify_clique_behavior`` or
``classify_clique_behavior_with_clockwork_pair_search``.

The ordinary classifier's optional ``octahedron_retract_iterates=(3,)``
search inspects only its completely pared third clique iterate, rather than
automatically searching every iterate. Set ``octahedron_dimensions=(4,)``
to restrict targets to exactly ``O_4``; budget each chosen iterate with
``octahedron_max_patterns``, ``octahedron_max_states``, and
``octahedron_max_assignments``. The default empty iterate tuple
performs **no** octahedral retraction search. Every positive witness is
independently replayed; reaching a resource limit is inconclusive.

Positive-iterate retraction searches require a complete labelled prefix and
its origin graph. Use ``verify_clockwork_retraction_certificate_detailed``
for independent replay with optional clique-count and edge bounds; its
``VERIFIED``, ``INVALID``, and ``LIMIT_REACHED`` outcomes distinguish a
failed proof from replay that could not finish. The Boolean verifier remains
available for unrestricted replay.

``verify_clockwork_pair_map_certificate`` independently checks Theorem 2.6
clockwork-pair certificates, including the canonical theorem source,
coaffinations, equivariance, graph-morphism edges (which may collapse), and
target provenance. For the combined classifier, pass the original graph;
the certificate records whether the searched base was that graph or its
completely pared graph. The clique-graph variant verifies unpared
``K(theorem_base_graph)`` and accepts the same replay limits.

.. automodule:: pycliques.clockwork_search
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Clockwork Pairs
---------------

.. automodule:: pycliques.clockwork_pairs
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Coaffinations
-------------

.. automodule:: pycliques.coaffinations
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Cutpoints
---------

.. automodule:: pycliques.cutpoints
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Dominated
---------

.. automodule:: pycliques.dominated
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Helly
-----

.. automodule:: pycliques.helly
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Helly Edge Covers
-----------------

.. automodule:: pycliques.helly_cover
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Homotopy Invariance
-------------------

.. automodule:: pycliques.homotopy_invariance
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Poset Homotopy
--------------

.. automodule:: pycliques.poset_homotopy
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Octahedral Boolean Model
------------------------

.. automodule:: pycliques.octahedral_boolean
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Homotopy Research Certificates
------------------------------

.. automodule:: pycliques.homotopy_research
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Named
-----

.. automodule:: pycliques.named
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Retractions
-----------

.. automodule:: pycliques.retractions
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Small
-----

.. automodule:: pycliques.small
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Surfaces
--------

.. automodule:: pycliques.surfaces
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Closed Neckties
---------------

The bitset index computes the intersection of maximal completes extending
an input complete without enumerating those maximal completes. The incidence
patterns also give exact center tests. Use incidence-preserving permutation
generators to reduce the enumeration to vertex-orbit representatives. A
failed center condition does not establish a failed homotopy equivalence.

For large clique families, use ``ClosedNecktieIndex(family, lazy=True)``
or construct an ``ImplicitCliqueGraph.from_family(family)`` and pass it to
``ClosedNecktieIndex.from_implicit(graph)``. The latter shares incidence
masks and preserves member IDs; the input graph must be unpared. The lazy
index stores only a bounded cache of neighborhoods, rather than a dense
neighborhood matrix. Family members are decoded on demand.

The closure computation eliminates a candidate and all its nonneighbors
in the original common neighborhood when a nonneighbor is found. Neither
can be universal there. This works independently of the storage mode.

The ``closed_containing`` and ``closed_representatives`` methods accept
the same interval-pruning callback as ``closed_within``. When starting from
a vertex, the upper family may not itself be a complete. A uniform
strong-collapse certificate remains valid for such an interval.

For applications that only require bad closed completes, use
``bad_closed_containing(anchor)``. A good closed family has a common
vertex, and any bad extension must include a member that misses it.
The method branches only over such members, retaining all other
candidates for later extensions. This covers every bad closed family
exactly once without symmetry.

Optional incidence-preserving ``stabilizer_generators`` must fix the
anchor. At each node the method retains a subgroup preserving both
the closed lower family and its permitted upper family. It chooses
an orbit of common incidence stars and branches on orbit
representatives of the members outside their intersection. It
excludes all earlier members of the chosen branch domain, not all
earlier indices. The resulting search covers every orbit of bad closed
families, possibly with repetitions. The same interval-pruning callback
is supported. Using only supplied generators that preserve the state
is conservative; it does not claim to compute the full stabilizer.

.. automodule:: pycliques.closed_neckties
   :members:
   :show-inheritance:
   :member-order: bysource

CliqueSequence: pared iterates and reproducible labels
-----------------------------------------------------

Use ``CliqueSequence(G, bound=10000, pare=True, relabel=True,
pare_initial=True)`` to construct completely pared iterates from an
arbitrary starting graph. The default still assumes the supplied graph
is already pared and retains its labels. For exact (unpared) iterates
use ``pare=False``. Deterministic integer relabeling is not isomorphism
canonization.

For positive level n, ``seq.clique_family(n)`` returns all maximal cliques
of the preceding output, even those subsequently removed by paring.
It does not construct the edges of level n. ``seq.raw_iterate(n)``
constructs the graph before paring. ``seq.raw_label_map(n)`` and
``seq.label_map(n)`` map raw and surviving labels, respectively, to cliques
of the preceding output (to the original input labels at level zero).
``seq.paring_certificate(n)`` records ordered deletions in raw labels.
A successful replay verifies domination in the current induced subgraph
at each deletion; compacting surviving integer labels is a separate
isomorphism.

The order and edge bounds are applied to the raw graph, not the pared
output. A clique family can remain available even when the edge bound
prevents materializing that level. Treat cached graphs as read-only.

For homotopy investigations, write P_0 for the completely pared starting
graph and P_{n+1} for the completely pared clique graph of P_n. Dominated
vertex deletion preserves homotopy type and induces a dismantling
reduction on clique graphs, so each P_n has the homotopy type of the
corresponding unpared iterate. Neckties and other incidence-level
properties of the actual iterates need not be preserved. A certificate
for a particular nerve comparison should therefore use the raw graph
and the clique family with the same labels.

.. code-block:: python

   import networkx as nx
   from pycliques import CliqueSequence, dominates

   seq = CliqueSequence(nx.path_graph(5), relabel=True, pare_initial=True)
   raw = seq.raw_iterate(0).copy()
   for v, w in seq.paring_certificate(0):
       assert v != w and dominates(raw, w, v)
       raw.remove_node(v)
   assert nx.is_isomorphic(raw, seq[0])
   assert seq[0].order() == 1

For completeness, the dismantling reduction is local: if v is dominated
by w in G, each maximal clique q containing v also contains w. If q
without v is not maximal in G without v, extend it to a maximal clique
p of G without v. Then p is also maximal in G, does not contain v, and
dominates q as a vertex of K(G), since every clique meeting q also meets
q without v. Delete all such q. The surviving cliques correspond
bijectively to the maximal cliques of G without v by removing v, and
their intersection relation is unchanged. Iterating proves the claim
for complete paring and all subsequent iterates.
