Triangular-cover divergence certificates
=========================================

``pycombtop.covering_divergence`` verifies a finite proof that a clockwork
pair-map certificate into ``K(B)`` transfers through a two-sheeted triangular
cover ``B -> G``. The verifier checks the full projection, its local
closed-neighborhood isomorphisms, the deck involution and displacement, the
induced action on ``K(B)``, and the inner Theorem 2.6 certificate. A clique
vertex or edge budget that prevents reconstruction returns
``LIMIT_REACHED``; malformed proof data returns ``INVALID``.

The opt-in ``pycombtop.covering_divergence_search`` module enumerates
connected double covers from normalized triangle-flat ``F_2`` edge cocycles.
It then checks the deck displacement, constructs the bounded clique graph,
and searches coaffination-orbit assignments for ``R_{2m}^n`` using explicit
state and assignment budgets:

.. code-block:: python

   from pycombtop import search_covering_divergence

   result = search_covering_divergence(graph, max_m=3, max_n=3)

``FOUND`` includes a replay-verified composite certificate.
``EXHAUSTED`` means the enumerated nonzero cohomology classes and configured
clockwork parameter range were searched completely. ``LIMIT_REACHED`` means
that cover-class, graph-size, clique-construction, or morphism limits left
work incomplete. Neither a bounded miss nor ``EXHAUSTED`` outside that
explicitly configured family is evidence of clique convergence. This search
is a separate research API and is not part of the ordinary classifier.

The regression case ``IEjdnpy^O`` has a one-dimensional ``H^1`` over
``F_2``. Its unique nonzero class produces a connected 20-vertex, 54-edge
triangular cover with deck displacement four. A verified equivariant map from
``R_6^0`` into the 26-vertex, 96-edge clique graph supplies the finite
Theorem 2.6 certificate. The composite verifier independently checks the
covering transfer; induced clockwork containment alone is not used as a
divergence proof.

The mathematical implications use Proposition 2.4, Theorems 2.6 and 3.1,
and Proposition 4.2 of:

F. Larrion, V. Neumann-Lara, and M. A. Pizana, "Graph Relations, Clique
Divergence and Surface Triangulations," *Journal of Graph Theory* 51 (2006),
110--122, DOI: `10.1002/jgt.20126 <https://doi.org/10.1002/jgt.20126>`_.

The subgroup-covering implementation follows S. Rees and L. H. Soicher, "An
algorithmic approach to fundamental groups and covers of combinatorial cell
complexes," *Journal of Symbolic Computation* 29 (2000), 59--77.
