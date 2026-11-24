pycombtop
=========

.. automodule:: pycombtop
   :no-members:
   :show-inheritance:

Simplex
-------

.. automodule:: pycombtop.simplex
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

S-Collapses
-----------

.. automodule:: pycombtop.s_collapses
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Homotopy Type
-------------

.. automodule:: pycombtop.homotopy_type
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Fundamental Group
-----------------

.. automodule:: pycombtop.fundamental_group
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Relative Collapse
-----------------

.. automodule:: pycombtop.relative_collapse
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Example: verifying a collapse certificate
------------------------------------------

This small graph has a relative elementary-collapse sequence from
:math:`X = \Delta(K(G))` to :math:`N(G)`. ``PROVED`` means that the sequence was replayed
successfully; ``INCONCLUSIVE`` means that the bounded search did not establish
a collapse. A verified collapse certificate is stronger than a homotopy
equivalence claim, while vanishing rational relative Betti numbers are only a
necessary condition and do not detect integral torsion.

.. code-block:: python

   import networkx as nx
   from pycliques import clique_graph
   from pycombtop import (
       CollapseStatus,
       clique_complex,
       nerve_of_cliques,
       relative_collapse,
       verify_relative_collapse,
   )

   G = nx.from_graph6_bytes(b"EElw")
   K = clique_graph(G)
   assert K is not None
   X = clique_complex(K)
   N = nerve_of_cliques(G)
   result = relative_collapse(X, N)
   assert result.status is CollapseStatus.PROVED
   assert verify_relative_collapse(X, N, result.pairs)

Indexed Relative Collapse
-------------------------

The greedy indexed engine enumerates only faces outside the preserved
subcomplex, and replays any complete certificate using direct coface tests.
It can be used before the original backtracking engine; a greedy dead end
or a relative-face budget exit is inconclusive.

.. automodule:: pycombtop.indexed_relative_collapse
   :members:
   :show-inheritance:
   :member-order: bysource


Uniform strong collapses
------------------------

A failed uniform-collapse search can still yield useful dominated-vertex
deletions. ``uniform_strong_collapse_reduction`` retains these moves
and an optional full cone certificate; ``verify_uniform_strong_collapse_prefix``
independently replays the moves without requiring a cone. If a prefix is
valid on an interval [M,E], it is valid on any nested interval: every
intermediate family of the latter was also in the former. In an individual
family, skip deletions whose removed vertex is absent. Restricting the
nested interval to the surviving vertices allows the search to resume
there. A partial prefix alone never proves contractibility.

.. automodule:: pycombtop.strong_collapses
   :members:
   :undoc-members:

Closed-trace contractibility
----------------------------

Use ``check_closed_trace_contractibility(index, strategy="vertex")`` to
search through vertex-orbit representatives directly, without first
enumerating maximal completes. The default ``strategy="maximal"`` retains
the existing maximal-complete-orbit strategy. The vertex strategy is
particularly useful for lazy indices of very large clique graphs. It may
still require many search nodes; a limit or unresolved trace is
inconclusive, not a counterexample to homotopy equivalence.

The optional ``strategy="bad_vertex"`` branches adaptively using
common incidence stars. A callback ``anchor_stabilizers(anchor)``
can supply incidence-preserving member permutations fixing each
anchor; these are validated by the index. The checker counts good
intermediate nodes toward its limit but need not test their
contractibility, since they have a common vertex. Bad nodes and
pruned intervals still require independently replayed certificates.

With ``inherit_prefixes=True``, reuse independently verified partial
collapse prefixes on nested intervals. The search checks both lower and
upper containment before reusing a projection, and resets its history
between orbit searches. Each successful suffix is independently replayed
on the projected family. The result records the number of inherited
checks. The default remains the original non-inheriting strategy.

.. automodule:: pycombtop.closed_trace_contractibility
   :members:
   :undoc-members:
