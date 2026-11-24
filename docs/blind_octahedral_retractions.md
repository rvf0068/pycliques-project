# Blind octahedral retractions: generic API

These functions search for retractions onto induced octahedra
\(O_d=\overline{dK_2}\) **without any graph-specific antipodal-pair presets**.
They accept arbitrary finite, simple, loopless NetworkX graphs, including
noninteger vertex labels.

## Quick example

```python
import networkx as nx
import pycliques

G = nx.complete_multipartite_graph(2, 2, 2, 2)
result = pycliques.find_blind_octahedral_retraction(
    G, target_dimension=4, beam=8, samples=12, delete_count=0,
)
if result.status is pycliques.OctahedralBlindStatus.FOUND:
    pairs = result.antipodal_pairs
    assert pairs is not None and result.projection is not None
    T = G.subgraph([v for pair in pairs for v in pair])
    assert pycliques.verify_induced_octahedron(
        G, pairs, require_dominating=True
    )
    assert pycliques.verify_retraction_witness(
        G, T, (result.projection, {v: v for pair in pairs for v in pair})
    )
```

The top-level heuristic returns `LIMIT_REACHED` for any failure; it **never
proves nonexistence**. The graph's vertex iteration order is relevant to
randomized reproducibility even with a fixed seed.

For conflict-guided repair, `max_repairs` bounds how many best-ranked
candidates are retained and passed to the completion solver, while
`max_candidate_checks` (default 200,000) independently bounds how many
deletion combinations are examined. Candidate generation is streamed and
uses a bounded heap, rather than materializing all combinations. A budget
cutoff is inconclusive and can miss a better repair; increasing the budget
may change the chosen candidates. The default covers all
`binom(22, 7) = 170544` deletions used in the original snub experiment.


## Exact tests and completion

```python
P = [(0, 1), (2, 3)]
screen = pycliques.screen_partial_octahedral_retraction(G, P)
# screen.possible=False proves no retracting octahedral extension of P.

completion = pycliques.complete_dominating_octahedron(
    G, P, target_dimension=4, max_states=10000
)
# FOUND includes checked target pairs; EXHAUSTED is a rigorous negative
# answer for extensions retaining P; LIMIT_REACHED is inconclusive.
```

During `complete_dominating_octahedron`, an additional **exact edge-pair
covering screen** discards branches that cannot be completed. If `T` is the
set still uncovered in the complement `H` and `r` antipodal pairs remain,
then any completion must satisfy, for every nonnegative integer weighting,

\[
\sum_{v\in T}w_v \leq r\max_{ab\in E(H[T])}
\sum_{v\in T\cap(N_H(a)\cup N_H(b))}w_v.
\]

The implementation uses unit weights and two inexpensive degree-based
integer weightings, restricted to small remaining subproblems to control
search overhead. A strict violation is a proof that **no extension retaining
the supplied partial pairs** is possible. This is a pruning optimization,
not a claim that a resource-bounded global retraction search is complete.

## Iterative forced-antipodal-pair pruning

Let \(X\) be the host, \(H=\overline{X}\), and \(P\) a partial collection
of antipodal pairs. Write \(T\) for vertices with no \(H\)-neighbor among
the endpoints of \(P\). Every further antipodal pair belongs to \(H[T]\),
and the new endpoints must collectively dominate **every vertex of \(T\)**
by open \(H\)-neighborhoods.

If \(x\in T\) has exactly one neighbor \(u\) in \(H[T]\), then every
completion must select \(u\) as an endpoint. Let \(F\) contain all
such compulsory endpoints. Since the target pairs induce a matching in
\(H\), **\(H[F]\) must have maximum degree at most one**. Every edge
\(uv\in E(H[F])\) must itself be an antipodal pair. Selecting all these
forced pairs and replacing \(T\) by
\(T\setminus(N_H(u)\cup N_H(v))\) preserves every possible completion.
Repeat until no more pairs are forced. An isolated undominated vertex,
or a vertex of degree at least two in \(H[F]\), proves that no completion
retaining \(P\) can exist.

The routine `pycliques.octahedral_forced_pairs.screen_forced_pairs`
implements this exact necessary test using complement bitmasks.
`screen_partial_octahedral_retraction` applies it **after** its
extension-safe frozen-domain 2-SAT test; the completion solver uses
forced pairs as compulsory recursive steps. For predictable costs,
the screen is applied only when at most 512 residual vertices remain.
A positive result remains inconclusive. The number of remaining pairs
also provides an exact bound: if more pairs are compulsory than the
requested remaining target dimension, no such completion exists.

This is a *vertex*-dominating induced-matching condition in \(H\).
It must not be confused with the literature definition of a dominating
induced matching (also called efficient edge domination), which requires
edge domination, not vertex domination. The two conditions differ in general.

The forced-subgraph screen solves a 2-SAT *necessary* condition among vertices
that cannot map to future antipodal pairs. It is not a sufficient condition
for a retraction. If the screen rejects a partial target, you can call
`minimize_octahedral_conflict` to remove unnecessary pairs. A reported
`deletion_minimal=True` means no single remaining pair can be deleted while
retaining this particular 2-SAT obstruction.

Given a conflict hypergraph on a **fixed list of antipodal pairs**, encode
its subsets as bit masks and compute a minimum-weight hitting set:

```python
cores = [0b0101, 0b0110, 0b1100]
repair = pycliques.minimum_octahedral_repair(cores, number_of_pairs=4)
print(repair.deleted_indices, repair.cost)
```

A hitting set is a lower bound on necessary changes, not a successful
completion certificate. `discover_partial_octahedral_fibers` is the separate
heuristic that constructs partial induced octahedra and anticomplete
retraction-fiber pairs simultaneously.

## Full snub reproduction

The reproducibility script `research/snub_blind_o24_from_api.py` reconstructs
\(K^4(S)\) with `pycliques.clique_graph`, uses the public generic API,
independently verifies every edge of the resulting retraction, and writes a
JSON certificate only after verification. No target pairs or existing
certificate are loaded for discovery.

```bash
uv run python research/snub_blind_o24_from_api.py \
    --output snub_blind_o24_reproduction.json
```

To opt into the expensive research regression test:

```bash
uv run pytest packages/pycliques/tests/test_blind_snub_regression.py -m slow
```

The historical fixed-target and new blind certificates can be checked
quickly with the ordinary non-slow regression tests. The blind search is
intentionally opt-in because its performance depends on graph ordering,
random seed and search budgets.

The mathematical foundations, proofs, and figures are in
[`blind_octahedral_retraction.tex`](blind_octahedral_retraction.tex).
The special-retraction characterization in Larrión--Pizaña--Villarroel-Flores,
*Ars Combinatoria* **142** (2019), Theorem 3.7, is distinct from the general
retractions constructed here.
