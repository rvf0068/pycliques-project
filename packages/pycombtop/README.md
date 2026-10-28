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

Order-7 and order-8 census JSON outputs are generated locally and are not
tracked in the repository. Reproduce the connected, labelled graph datasets
(considered up to isomorphism by the bundled graph6 data) with:

```sh
uv run python research/homotopy_census.py 7 --max-states 250 --max-simplices 500 --max-clique-graph-edges 100000
uv run python research/homotopy_census.py 8 --max-states 250 --max-simplices 500 --max-clique-graph-edges 100000
```

These process 853 and 11,117 connected graphs, respectively. The graph6 data
identifies each isomorphism class; `connected=True` is the census assumption.
Run `make research-test` for fast regressions including the order-7 candidate
count. The order-8 candidate count is a slower regression available through
`make research-test-slow`; CI runs it on schedule and manual dispatch.

For larger runs, stream JSONL records and resume safely with unchanged
parameters:

```sh
uv run python research/homotopy_census.py 8 --jsonl --output census-order8.jsonl \
  --max-states 250 --max-simplices 500 --max-clique-graph-edges 100000
uv run python research/homotopy_census.py 8 --jsonl --resume \
  --output census-order8.jsonl --max-states 250 --max-simplices 500 \
  --max-clique-graph-edges 100000
```

The JSONL header records Python, NetworkX, and package versions plus the order
and budgets. A separate `.summary.json` file is written when the dataset
finishes; resume rejects mismatched metadata and duplicate record indices.
The summary distinguishes a fully enumerated dataset from candidate-filter
results that were inconclusive for individual graphs.
