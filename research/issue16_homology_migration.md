# Issue #16: exact finite-field homology backend and historical audit

This report records the completed repair and the **read-only** validation of
the previously published homotopy-census output. Historical verdict files
and their original dependency manifests are intentionally unchanged.

## Root cause and replacement

The former production Betti-number path was
`pycombtop.homotopy_type.betti_numbers_graph` and
`betti_numbers_sc`, which delegated to MoguTDA's floating-point
matrix-rank calculation. On the snub-disphenoid clique graph and its
second iterate, those ranks disagreed with independently exact
boundary-matrix calculations. In particular, `K(S)` and `K²(S)`
have only one nonzero reduced mod-2 Betti number: beta_2=1.

The replacement uses **GUDHI 3.13.0** with explicit coefficient field 2.
For general simplicial complexes, maximal facets are inserted directly,
never inferred by expanding a one-skeleton. Exhausting the configured face
budget raises `HomologyResourceLimit`; the homotopy classifier marks
these results limited and never assigns an exact wedge multiplicity.

The separately maintained `bounded_reduced_betti_mod2` reducer uses
integer-bitset GF(2) boundary matrices and remains an independent verifier.
The dependency lockfile no longer includes MoguTDA.

Note: mod-2 Betti numbers do not determine an arbitrary homotopy type.
They determine the sphere multiplicities *only after* an independent
homotopy certificate (such as verified vertex decomposability) proves
that the complex is a wedge of spheres.

## Historical homotopy-census audit

Source: [rvf0068/pycliques-homotopy-census](https://github.com/rvf0068/pycliques-homotopy-census),
repository HEAD at audit time `40f486903b60f34ee16b2f8aaea7545883be54b4`.

All saved compressed verdict chunks of orders 6 through 10 were scanned.
A record is flagged when its exact homotopy explanation contains the
historic nonpure vertex-decomposability proof whose sphere counts came
from the old homology calculation. Inexact `BETTI_ONLY` records were
also flagged. The audit recomputed **every** flagged profile using both
the GUDHI production implementation and independent exact GF(2) boundary
reduction, checking agreement with the saved result.

| Census source | Flagged verdicts | Exact rechecks | Mismatches |
|---|---:|---:|---:|
| Order 8, K(G), exact vertex-decomposability | 1 | 1 | 0 |
| Order 9, K(G), exact vertex-decomposability | 100 | 100 | 0 |
| Order 10, G, exact vertex-decomposability | 145 | 145 | 0 |
| Order 10, K(G), exact vertex-decomposability | 36 | 36 | 0 |
| Order 10, G, inexact BETTI_ONLY | 1 | 1 | 0 |
| **Total** | **283** | **283** | **0** |

Order 10 contained 11,716,571 connected graphs; order 9 contained
261,080. No incomplete exact rechecks occurred, and all other
historical exact verdicts used proof paths not directly dependent
on this legacy Betti-number function.

The primary [full audit workflow](https://github.com/rvf0068/pycliques-project/actions/runs/44404505988)
passed and saved the JSON record inventory and zero-difference
verification report as
[artifact 14473468493](https://github.com/rvf0068/pycliques-project/actions/runs/44404505988/artifacts/14473468493).

## Reproduction

```bash
uv sync --frozen --python 3.11
uv run pytest packages/pycombtop/tests/test_gudhi_homology_issue16.py
git clone --depth 1 --filter=blob:none \
  https://github.com/rvf0068/pycliques-homotopy-census.git ../pycliques-homotopy-census
uv run python research/audit_issue16_homotopy_census.py \
  --root ../pycliques-homotopy-census --orders 6 7 8 9 10 \
  --recheck 1000 --output issue16-audit.json
```

The audit script is read-only. Its `--recheck 1000` ceiling exceeds
every affected category, and the output's
`unverified_count_lower_bound` was **zero**.

## Scope of the conclusion

This audit eliminates the specifically identified MoguTDA-dependency
risk for the saved census verdicts. It is not a general independent
reverification of every algorithmic homotopy certificate in the census,
nor a proof of snub-disphenoid clique convergence or divergence.
