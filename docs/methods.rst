Choosing a method
=================

Finding an induced octahedron
-----------------------------

Use ``find_induced_octahedron_amplified`` or
``find_induced_octahedron_compatibility``. The returned antipodal pairs
certify an induced octahedral subgraph. A search budget can leave the
embedding question unresolved.

- `Compatibility graphs and antipodal pairs <https://github.com/rvf0068/pycliques-project/blob/main/docs/blind_octahedral_compatibility.md>`_
- `General embedding and amplification <https://github.com/rvf0068/pycliques-project/blob/main/docs/general_octahedral_amplification.md>`_
- `Implicit clique-graph amplification <https://github.com/rvf0068/pycliques-project/blob/main/docs/implicit_octahedral_amplification.md>`_
- `Amplifying small octahedral seeds <https://github.com/rvf0068/pycliques-project/blob/main/docs/octahedral_amplification.md>`_
- `MILP embedding search <https://github.com/rvf0068/pycliques-project/blob/main/docs/octahedral_ilp.md>`_

Retracting onto specified antipodal pairs
-----------------------------------------

Use ``find_octahedral_retraction`` or ``find_octahedral_retraction_shores``
for a chosen embedding. Check the projection and inclusion with
``verify_retraction_witness``. Exhaustion applies to these particular pairs.

- `Retraction and implicit clique-graph operations <https://github.com/rvf0068/pycliques-project/blob/main/docs/general_retraction_and_implicit_cliques.md>`_
- `Blind search and independent replay <https://github.com/rvf0068/pycliques-project/blob/main/docs/blind_octahedral_retractions.md>`_
- `The snub disphenoid and its seeded O24 retraction <https://github.com/rvf0068/pycliques-project/blob/main/docs/snub_o24_retraction.md>`_

Looking for any octahedral retract
----------------------------------

``find_octahedral_retraction_coaffine`` searches over targets. Inspect
the returned status and searched dimension range before interpreting
exhaustion. The Python verifier replays a positive native witness.

- `Native search and the cubic 379 replay <https://github.com/rvf0068/pycliques-project/blob/main/docs/coaffine_native.md>`_
- `Unseeded coaffine search <https://github.com/rvf0068/pycliques-project/blob/main/docs/coaffine_octahedral_search.md>`_
- `Homology-guided retractions on pared iterates <https://github.com/rvf0068/pycliques-project/blob/main/docs/iterated_homological_retractions.md>`_

Interpreting and replaying a classification
-------------------------------------------

Pass ``references=select_reference_graphs(proved_only=True)`` for a
classification using proved registered references. Recursive rules inherit the
selection. The first successful rule still determines the saved reason.
``pycliques.certificate_records.serialize_result`` writes a schema-3 record
for supported Helly, special-octahedron and retraction proofs.
``verify_record(record)`` checks the witness and rebuilds the pared iterate.

``retraction(G,H)`` now yields each projection once per inclusion selected up
to host and target automorphisms. Historical yield counts may include duplicates.
``first_homology_rank(G)`` computes the free rank of the abelianization.
Use ``fundamental_group(G).generator_count`` to index a presentation's generators.

- `Classifier and two-stage census <https://github.com/rvf0068/pycliques-project/blob/main/docs/clique_behavior.md>`_
- `Certified homotopy reductions <https://github.com/rvf0068/pycliques-project/blob/main/docs/certified_homotopy.md>`_

Historical census workers
-------------------------

The downstream v3 timeboxed worker, serializer and upstream revision are
identified by SHA in its run manifest. Replaying those files requires their
original versions. For new runs, use the public reference selector and witness
format above. Existing indeterminate-file columns and schema-3 proofs remain
readable.
