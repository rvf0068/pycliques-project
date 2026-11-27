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


Finite posets and LPV08 constructions
-------------------------------------

The finite-poset interface stores a partial order independently of its
order complex. Strict generating pairs are oriented from lower to upper;
isolated elements are supplied separately. The relation constructor checks
transitivity, and the matrix constructor requires a reflexive matrix.
There is no new dependency.

For the barycentric subdivision of a complex, use
``order_complex(face_poset(complex_))``. The face poset contains only actual
nonempty faces, not ghost vertices or the empty face. Construction of an
order complex enumerates maximal chains, not every chain. Materializing
face and interval posets can still be expensive: the order uses space
proportional to the number of comparable pairs, and a face poset may have
exponentially many elements.

The following reproduces the graph constructions of
Larrión, Pizaña and Villarroel-Flores, *Posets, clique graphs and their
homotopy type*, European Journal of Combinatorics 29 (2008), 334–342,
https://doi.org/10.1016/j.ejc.2006.04.012.

.. code-block:: python

   import networkx as nx
   from pycliques import clique_graph, homotopy_clique_graph
   from pycombtop import (
       beat_core,
       complete_poset,
       interval_poset,
       is_up_helly,
       mho,
       omega,
       order_complex,
       verify_beat_reduction,
   )

   G = nx.cycle_graph(4)
   P = complete_poset(G)
   assert nx.is_isomorphic(omega(P), G)
   assert nx.is_isomorphic(mho(P), clique_graph(G))
   assert nx.is_isomorphic(mho(interval_poset(P)), homotopy_clique_graph(G))
   assert is_up_helly(P)
   X = order_complex(P)
   reduction = beat_core(P)
   assert verify_beat_reduction(P, reduction)

The last assertion verifies a finite-space beat-point core, **not** a
classification of the homotopy type of the order complex. A one-point
beat core proves contractibility, but a larger core need not imply
noncontractibility of the order complex. Use the existing simplicial
homology, homotopy and discrete-Morse tools on ``X`` for such questions.

Other constructions include bounded complexes and their face posets,
the two Helly and Sperner conditions, atomic and coatomic projections,
join-increasing recognition, products, ideals, crosscuts, intervals,
conical contraction witnesses and Quillen lower fibers. The functions
``nice_fibers`` and ``lower_fibers`` return the actual subposets to be
tested; they do not declare a homotopy equivalence on the basis of
vanishing Betti numbers. The LPV08 conditions on bounded subsets are
understood for nonempty subsets, as is appropriate for face posets.

For graphs use the existing optimized ``pycliques.double_bound_graph(G)``
and ``pycliques.homotopy_clique_graph(G)`` rather than materializing
``complete_poset(G)`` only to construct those graphs. The general
``pycombtop.double_bound_graph(P)`` accepts a poset instead. Graph nodes
in the general constructions retain their poset labels (intervals are
endpoint pairs); NetworkX does not allow None as a node label. The
specialized closed-trace checker in ``pycombtop`` remains the preferred
implementation of the LPV08 Theorem 5.7 sufficient hypothesis for graphs.

.. automodule:: pycombtop.posets
   :members:
   :show-inheritance:
   :member-order: bysource


Named posets
------------

All named constructors return the same ``Poset`` type, so the existing
order-complex, finite-space and clique-graph tools apply without adapters.
The Boolean lattice uses subset cardinality as rank; the partition lattice
uses the number of elements minus the number of blocks. Both support
inclusive rank intervals and removal of the original minimum and maximum.
The proper part of the Boolean lattice on four elements has the homotopy
type of a 2-sphere; the proper part of the partition lattice on four
elements has the homotopy type of a wedge of six circles.

.. code-block:: python

   from pycombtop import (
       betti_numbers_sc,
       boolean_lattice,
       order_complex,
       partition_lattice,
   )

   B = boolean_lattice(4, proper=True)
   assert betti_numbers_sc(order_complex(B)) == [0, 0, 1]
   Pi = partition_lattice(4, proper=True)
   assert betti_numbers_sc(order_complex(Pi)) == [0, 6]

The reusable partition enumeration, refinement, indexed comparability
graph and induced symmetric-group permutation helpers formerly in
``research.partition_lattice`` now live in ``pycombtop.named_posets``.
The research module still provides its original imports and CLI.

.. automodule:: pycombtop.named_posets
   :members:
   :show-inheritance:
   :member-order: bysource

Subgroup lattices and p-subgroup posets
---------------------------------------

The group constructors accept a SymPy finite permutation group, including
the standard constructors for cyclic, abelian, dihedral, alternating and
symmetric groups. Each label is a frozenset of actual group elements
(SymPy permutations), so distinct conjugate subgroups remain distinct.
The order is subgroup inclusion. Brown's poset includes all nontrivial
p-subgroups; Quillen's poset includes only the nontrivial elementary
abelian p-subgroups. The trivial subgroup is excluded in both cases,
but the whole group is included when it satisfies the defining condition.
A non-prime p is rejected. A prime not dividing the group order produces
an empty poset. Conjugation induces order automorphisms, available via
``subgroup_conjugation``.

.. code-block:: python

   from sympy.combinatorics.named_groups import CyclicGroup, SymmetricGroup
   from pycombtop import (
       brown_poset,
       quillen_poset,
       subgroup_lattice,
       order_complex,
   )

   C4 = CyclicGroup(4)
   assert len(brown_poset(C4, 2)) == 2
   assert len(quillen_poset(C4, 2)) == 1

   S3 = SymmetricGroup(3)
   assert len(subgroup_lattice(S3)) == 6
   Q = quillen_poset(S3, 2)
   assert len(Q) == 3
   X = order_complex(Q)

Enumeration extends generating sets exhaustively and can restrict the
search to p-groups or elementary abelian p-groups. It is not a conjugacy
class enumeration. The default bounds are group order 1000 and 10000
enumerated subgroups (including the trivial subgroup). Set either bound
to None to disable it. An exceeded bound raises
``SubgroupEnumerationLimit`` rather than returning an incomplete poset.
Even within these bounds, time and storage costs can grow substantially.
Finitely presented groups are not accepted directly: a finite permutation
representation can be supplied instead.

.. automodule:: pycombtop.subgroup_posets
   :members:
   :show-inheritance:
   :member-order: bysource


Graph multihomomorphism posets
------------------------------

For undirected graphs (loops allowed), Dochtermann's Hom poset consists
of nonempty set-valued vertex maps with every source edge mapped to a
complete cross-product of target edges. The order is pointwise inclusion.
See A. Dochtermann, *Hom complexes and homotopy theory in the category of
graphs*, European Journal of Combinatorics 30 (2009), 490-509,
https://doi.org/10.1016/j.ejc.2008.04.009.

Use ``hom_poset(G, H)`` to construct the poset, and
``hom_order_complex(G, H)`` for its order complex (the barycentric
subdivision of the polyhedral Hom complex). The existing
``hom_graph(G, H)`` is instead the compatibility graph on ordinary
homomorphisms. Its clique complex is not in general the order complex
itself, though it is homotopy equivalent by the nerve lemma.

.. code-block:: python

   import networkx as nx
   from pycombtop import hom_poset, order_complex, betti_numbers_sc

   P = hom_poset(nx.complete_graph(2), nx.complete_graph(3))
   assert len(P) == 12
   assert len(P.minimal_elements()) == 6
   assert len(P.maximal_elements()) == 6
   assert betti_numbers_sc(order_complex(P)) == [0, 1]
   C = P.comparability_graph()
   assert nx.is_isomorphic(C, nx.cycle_graph(12))

Each poset element is a frozenset of (vertex, frozenset of images)
pairs; ``dict(eta)`` recovers the set-valued map. Ordinary
homomorphisms correspond precisely to the minimal elements. The
shared backtracking engine preserves the edge convention, including
loops. Adjacent vertices can have equal images only when the
corresponding target vertex has a loop.

The default explicit-poset limit is 10000 elements. Exceeding it raises
``HomPosetLimit``, rather than returning an incomplete poset. The
iterator ``iter_multihomomorphisms`` does not materialize the order
and can be used to inspect individual maps. Empty domain gives one
empty mapping; a nonempty domain and empty target give no maps.

The related general bridges already exist as ``face_poset(X)``,
``complete_poset(G)``, ``order_complex(P)``, and
``P.comparability_graph()``. In particular, the comparability graph
has the order complex as its clique complex.

.. automodule:: pycombtop.hom_complex
   :members:
   :show-inheritance:
   :member-order: bysource
