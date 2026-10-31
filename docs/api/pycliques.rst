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
``classify_clique_behavior_with_theorem_2_6``.

Positive-iterate retraction searches require a complete labelled prefix and
its origin graph. Use ``verify_clockwork_retraction_certificate_detailed``
for independent replay with optional clique-count and edge bounds; its
``VERIFIED``, ``INVALID``, and ``LIMIT_REACHED`` outcomes distinguish a
failed proof from replay that could not finish. The Boolean verifier remains
available for unrestricted replay.

``verify_theorem_2_6_certificate`` independently checks Theorem 2.6
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
