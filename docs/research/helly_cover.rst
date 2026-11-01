Helly Edge-Cover Gate Propagation
=================================

The :mod:`pycliques.helly_cover` module studies an explicitly supplied finite
family of complete vertex sets of a graph. This is different from asking
whether the graph's family of maximal cliques is clique-Helly.

``verify_helly_edge_cover`` independently checks that each selected member is
a complete, every graph edge is covered, and the selected family is Helly. A
failure includes either an uncovered edge or a pairwise-intersecting
subfamily with empty intersection.

``prune_helly_cover_candidates`` applies the gate characterization as a
necessary local condition to a finite candidate universe. It removes only
candidates that cannot belong to a Helly edge cover contained in the current
universe (subject to any explicitly supplied required facets, private edges,
and fixed gates). Its ``PRUNED_EXHAUSTIVE`` result means only that propagation
reached a fixed point. Surviving candidates do not constitute a selected
cover, and no existence solver is implied. ``NO_COVER_CERTIFIED`` is returned
only with a sound exclusion witness: a required facet was removed or an edge
is uncovered after propagation. If a work cap interrupts processing, the
result is ``INCONCLUSIVE_BUDGET`` and contains no negative witness.

Candidate labels are mapped to internal indices in graph iteration order.
The result includes this mapping and offers deterministic JSON output; label
representations in that output are descriptive metadata, not a way to rebuild
arbitrary Python objects. Research certificates that must be independently
replayed should define a concrete label encoding and reconstruct the graph
from it. Required facets, private edges, and fixed gate assignments are
serialized explicitly, so a negative result remains scoped to its stated
normalization assumptions.

The gate rule is a necessary-condition filter, not an unconditional conclusion
about clique-graph homotopy or the homotopy type of clique complexes.

O8 research certificates
------------------------

The opt-in driver
``research/o8_helly_cover_certificates.py`` generates a deterministic
rank-at-most-five normalized gate certificate and independently checks each
case with a separate set-based implementation:

.. code-block:: console

   uv run python research/o8_helly_cover_certificates.py \
     --max-rank 5 --output research/results/o8_helly_gate_certificate.json \
     --verify
   uv run python research/o8_helly_cover_certificates.py \
     --check research/results/o8_helly_gate_certificate.json

It refuses to overwrite an existing output file. The representative generator
exhaustively checks its canonicalization against all literal gate assignments.
Certificates include per-case propagation counts, a digest of sorted final
candidate masks, and either a missing required facet or an uncovered-edge
witness. The checker reconstructs the fixed graph and candidate universe and
replays every elimination; a digest alone is not treated as proof.
The normalized rank-at-most-five run has 7,168 representatives and reports
7,128 certified exclusions plus 40 surviving candidate universes in 8
symmetry types. Those 40 are not asserted to contain a Helly edge cover.
The rank-five exclusions rule out only their particular normalized gate
assignments; the 40 survivors prevent a global nonexistence conclusion.

The rank-four baseline covers the all-rank-four case after private-edge
normalization. A finite edge cover has an inclusion-minimal edge-covering
subfamily, in which every member has a private edge. In ``O8``, a private-edge
member of rank two or three is impossible by the gate condition: for rank two
choose a vertex adjacent to both endpoints, and for rank three choose the
antipode of its third vertex. This reduction and the rank-four exclusion do
not remove rank-five covers with one of the 40 surviving gate universes.

The separate rank-four private-edge baseline exhausts every locally possible
gate assignment without symmetry reduction (2,304 assignments):

.. code-block:: console

   uv run python research/o8_helly_cover_certificates.py \
     --four-uniform \
     --output research/results/o8_four_uniform_helly_certificate.json \
     --verify
   uv run python research/o8_helly_cover_certificates.py \
     --check research/results/o8_four_uniform_helly_certificate.json

These are finite, normalized search results. They do not prove existence of a
rank-five Helly cover when candidates survive, and do not imply a solution to
the clique-graph homotopy conjecture.
