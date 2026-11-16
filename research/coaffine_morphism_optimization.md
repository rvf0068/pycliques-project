# Constraint propagation for bounded coaffine graph morphisms

The library's private search `pycombtop.covering_divergence_search._bounded_pair_morphism`
now represents source coaffination orbits as constraint variables. Each
variable has a finite target-domain bit set; a choice for one orbit's
representative fixes the image of its involutive mate. The optimized
algorithm performs:

1. Unary filtering for coaffination-fixed vertices and edges internal
   to a two-vertex coaffination orbit.
2. Explicit edge-compatibility masks between different source orbits.
   **Equality of target images is permitted**, so this is a graph
   morphism search rather than an induced-subgraph embedding problem.
3. Arc consistency (AC-3), deterministic minimum-remaining-values
   search ordering and reversible constraint propagation on a trail.
4. Iterative (nonrecursive) backtracking and strict, separately
   counted maximum states and candidate assignments.

Every returned map is independently checked on all source edges and
the coaffination equations. A bounded failure never certifies
nonexistence outside the completed finite search family.

## Exact regression evidence

The solver is checked against a separate exhaustive map enumerator
on 80 deterministic small involutive graph pairs, and against the
published-cover/clockwork positive witness for `IEjdnpy^O`. The
high-orbit-count, malformed involution and resource-budget tests
remain in the `pycombtop` package.

## Reproducible nine-core benchmark

The eight completely pared nine-vertex core inputs are recorded
directly in `research/benchmark_coaffine_morphism.py`, with their
historical zero-based nine-vertex census indices. They come from the
frozen order-10 nine-core cross-reference in `pycliques-census`.
The following counts were obtained with the **same** resource and
clockwork family limits used by the previous census run:

| Nine-core representative | Historical candidate attempts | AC-3 attempts | Result |
|---:|---:|---:|---|
| 2317843 | 100,000 | 120 | EXHAUSTED |
| 3148541 | 100,000 | 108 | EXHAUSTED |
| 3505942 | 100,000 | 132 | EXHAUSTED |
| 4952669 | 100,000 | 96 | EXHAUSTED |
| 4953338 | 100,000 | 120 | EXHAUSTED |
| 5601875 | 100,000 | 114 | EXHAUSTED |
| 6938156 | 100,000 | 122 | EXHAUSTED |
| 7315343 | 100,000 | 132 | EXHAUSTED |

The original solver used every permitted assignment and returned
`LIMIT_REACHED` for all eight. The new algorithm exhausts the
same search family without needing larger budgets. The precise
candidate-attempt reduction is between approximately 758-fold and
1042-fold. These are **not** wall-clock speedup ratios.

The finite family consists of the unique nonzero triangular
double-cover cohomology class in each case and the eligible
clockwork source parameters \(m\in\{2,3\}\), \(n=0\), subject
to the recorded size and morphism limits. The result rules out
only certificates arising in this restricted family, **not**
all possible divergence mechanisms; no new clique-dynamics
classification follows from these negative searches.

Reproduce with:

```sh
uv run pytest packages/pycombtop/tests/test_covering_divergence_search.py -v
uv run python research/benchmark_coaffine_morphism.py --output /tmp/core9-ac3.json
```

An automated workflow additionally uploads the full eight-case
JSON benchmark. Historical `pycliques-census` stages use their
original pinned monorepo SHA and must not be retroactively changed.
After this library PR is merged, any updated census run should
be a new separately versioned, explicitly pinned experiment.
