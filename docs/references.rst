References
==========

These papers justify the *theorems and algorithms* used by the code. They do
not justify experimental classification outcomes (for example census
verdicts), which are the output of finite searches whose certificates are
checked independently.

Bibliography
------------

**[LNP06]** F. Larrión, V. Neumann-Lara, M. A. Pizaña, *Graph relations,
clique divergence and surface triangulations*, J. Graph Theory 51 (2006),
110–122. `doi:10.1002/jgt.20126 <https://doi.org/10.1002/jgt.20126>`_.
Theorem 2.6: an admissible relation between coaffine graphs transfers
rank divergence. Theorem 3.1: the clockwork graphs :math:`R_{2m}^n` are
rank divergent under the stated parameter restrictions.

**[LNP09]** F. Larrión, V. Neumann-Lara, M. A. Pizaña, *On expansive graphs*,
European J. Combin. 30 (2009), 372–379.
`doi:10.1016/j.ejc.2008.05.005 <https://doi.org/10.1016/j.ejc.2008.05.005>`_.
Theorem 4.6 (Neumann-Lara's connected summand theorem).

**[FANP13]** M. E. Frías-Armenta, F. Larrión, V. Neumann-Lara, M. A. Pizaña,
*Edge contraction and edge removal on iterated clique graphs*, Discrete
Appl. Math. 161 (2013), 1427–1439.
`doi:10.1016/j.dam.2013.02.003 <https://doi.org/10.1016/j.dam.2013.02.003>`_.
Theorem 2.1: clique retractions. Theorem 6.1: contraction of a local
bridge preserves clique behavior. Theorem 6.2: removing an edge that lies in
no triangle is ordered in clique behavior (not necessarily an equivalence).

**[LPV08]** F. Larrión, M. A. Pizaña, R. Villarroel-Flores, *Posets, clique
graphs and their homotopy type*, European J. Combin. 29 (2008), 334–342.
`doi:10.1016/j.ejc.2006.04.012 <https://doi.org/10.1016/j.ejc.2006.04.012>`_.
Theorem 5.7: if :math:`\Delta(h(C))` is contractible for every complete
:math:`C` of :math:`K(G)`, then :math:`G \simeq K(G)`. This is a
*sufficient* hypothesis only.

**[LPV09]** F. Larrión, M. A. Pizaña, R. Villarroel-Flores, *The fundamental
group of the clique graph*, European J. Combin. 30 (2009), 288–294.
`doi:10.1016/j.ejc.2007.12.006 <https://doi.org/10.1016/j.ejc.2007.12.006>`_.
Fundamental groups are preserved by the clique-graph operator; this does
not prove a general homotopy equivalence.

**[RS00]** S. Rees, L. H. Soicher, *An algorithmic approach to fundamental
groups and covers of combinatorial cell complexes*, J. Symbolic Comput. 29
(2000), 59–77.
`doi:10.1006/jsco.1999.0292 <https://doi.org/10.1006/jsco.1999.0292>`_.

Source-to-implementation map
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Implementation
     - Source
     - Notes
   * - ``pycliques.small.classify_clockwork_pair_map``,
       ``classify_clique_behavior_with_clockwork_pair_search``,
       ``verify_clockwork_pair_map_certificate``
     - [LNP06, Theorems 2.6 and 3.1]
     - Bounded search for an admissible map from a clockwork source
       :math:`R_{2m}^n` (with :math:`m \ge 2`) into the input graph or its
       clique graph. A verified certificate is an instance of a published
       sufficient criterion, not a new theorem.
   * - Certificate rule ``theorem_2_6_clockwork_pair_map``
     - [LNP06, Theorems 2.6 and 3.1]
     - Stored provenance string: the map goes into the (pared) input graph.
   * - Certificate rule ``theorem_2_6_clockwork_pair_map_clique_graph``
     - [LNP06, Theorems 2.6 and 3.1]
     - Stored provenance string: the map goes into :math:`K(G)`, giving rank
       divergence of :math:`K(G)` and hence clique divergence of :math:`G`.
       Rule strings are kept stable because saved census data may use them.
   * - ``pycliques.small`` suspension rule (``theorem_4_6_suspension_2_coaffination``)
     - [LNP09, Theorem 4.6]
     - Application to :math:`H * \overline{K_2}`, not the full theorem.
   * - ``pycliques.small`` local-bridge contraction
     - [FANP13, Theorem 6.1]
     -
   * - ``pycliques.small`` triangle-free edge removal and inverse cutpoint
       extension
     - [FANP13, Theorems 6.1 and 6.2]
     - Edge removal gives an ordered comparison only.
   * - ``pycliques.retractions``
     - [FANP13, Theorem 2.1]
     - Clique retractions.
   * - ``pycliques.poset_homotopy.check_double_bound_h_closure_contractibility``
     - [LPV08, Theorem 5.7]
     - Checks the sufficient hypothesis on :math:`K(DB(P(G)))`.
       ``COUNTEREXAMPLE`` rejects that hypothesis only; ``INCONCLUSIVE`` is
       neither a proof nor a counterexample.
   * - ``pycliques.homotopy_invariance.check_closed_clique_intersection_contractibility``
     - [LPV08, Theorem 5.7]
     - Repository test of the same kind of sufficient hypothesis over closed
       images; ``False`` does not disprove :math:`G \simeq K(G)`.
   * - ``pycliques.homotopy_invariance.necktie_center_condition_holds``
     - research notes
     - Repository-specific condition; not a published theorem statement.
   * - ``pycombtop.fundamental_group``
     - [RS00], [LPV09]
     - Coset enumeration and covers; fundamental-group context.
