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

.. automodule:: pycombtop.strong_collapses
   :members:
   :undoc-members:

Closed-trace contractibility
----------------------------

.. automodule:: pycombtop.closed_trace_contractibility
   :members:
   :undoc-members:
