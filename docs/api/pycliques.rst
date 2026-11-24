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
