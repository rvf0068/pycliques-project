# Homotopy and collapse terminology

The greedy `collapse` routine does not preserve a subcomplex, and failure to
find another free face is not a proof of non-collapsibility. For a checked
relative certificate, use `relative_collapse` and independently validate its
elementary pairs with `verify_relative_collapse`. A found elementary collapse
is stronger than homotopy equivalence: it gives simple-homotopy equivalence,
whereas homotopy equivalence alone need not arise from such a sequence.
Simple-homotopy equivalence can also use expansions, so it does not itself
provide a collapse from this source to this fixed target. An inconclusive
search only reports the limits of the chosen algorithm.

`relative_betti_numbers` computes ordinary relative homology over :math:`Q`
with exact ranks by default; rational ranks do not detect integral torsion.
Vanishing relative Betti numbers are necessary for a collapse, but are not a
collapse certificate.

The distinction matters in clique-graph research. See Islas-Gómez,
“Propiedades homotópicas del operador de clanes en gráficas,”
[Pädi 10 (2022), DOI 10.29057/icbi.v10iEspecial.8511](https://doi.org/10.29057/icbi.v10iEspecial.8511),
and Larrión, Neumann-Lara, and Pizaña, “On the homotopy type of the clique
graph,” *J. Braz. Comp. Soc.* 7 (2001), 69–73. The octahedron is an important
exception to unqualified Whitney-surface homotopy claims; finite computations
are evidence only for the graph orders and budgets actually completed.

The bounded research census is available as
`uv run python research/homotopy_census.py 7` (or order `8`). Its JSON records
the graph6 value and NetworkX version, candidate filters, defect, a verified
relative-collapse certificate when found, and exact rational relative Betti
numbers when within the simplex budget. Budget-limited searches are explicitly
reported as `INCONCLUSIVE`, never as non-collapsible.

The saved order-7 and order-8 census files were generated with
`max_states=250` and `max_simplices=500`; their records preserve those limits
and report all 853 and 11,117 connected graphs, respectively.