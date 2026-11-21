# Blind compatibility-based induced-octahedron search

An induced octahedron \(O_d=\overline{dK_2}\) in a graph \(G\) is specified
by \(d\) disjoint nonedges that are pairwise compatible: all four
cross-edges between each two nonedges belong to \(G\).

Equivalently, form a compatibility graph whose vertices are the nonedges
of \(G\), and join two compatible nonedges. A complete set of \(d\)
vertices in this auxiliary graph corresponds exactly to an induced
\(O_d\) in \(G\). **The implementation does not materialize this auxiliary
graph.** Instead, it intersects closed-neighborhood bitmasks, samples
compatible nonedges, and uses pair-replacement moves to reduce the
number of external common closed neighbors.

## Python API

```python
import pycliques

result = pycliques.find_induced_octahedron_compatibility(
    graph,
    target_dimension=20,
    max_restarts=500,
    local_steps=150,
    pair_samples=32,
    seed=24002,
    require_dominating=True,
)

if result.status is pycliques.OctahedronSearchStatus.FOUND:
    pairs = result.antipodal_pairs
    assert pairs is not None
    assert pycliques.verify_induced_octahedron(
        graph, pairs, require_dominating=True
    )
    # An embedding is not automatically a retraction.
    witness = pycliques.find_octahedral_retraction(
        graph, pairs, max_states=10000
    )
    if witness.status is pycliques.RetractionSearchStatus.FOUND:
        print("Certified octahedral retraction found")
```

The random seed is for sampling; it is **not** a list of known antipodal
pairs. No stored snub embedding or retraction projection is used. The
same seed can behave differently under a different vertex ordering.

Unlike the earlier nonlocal search, this method explicitly performs
fixed-dimension replacements, useful after a large nondominating
octahedron has been found.

## Blind snub benchmark

The regression test reconstructs \(K^4(F_{4,4})\) without using any
saved witness. It has 1,076 vertices and 504,360 edges. With
target_dimension=20, max_restarts=500, seed=24002, and
require_dominating=False, it checks blind discovery and independently
verifies an induced octahedron of dimension at least 20.

Complement domination is a separate and harder heuristic objective.
A local experiment produced a dominating \(O_{20}\) after 411 restarts,
but that outcome did not reproduce in GitHub Actions with the same
500-restart budget. Consequently the regression must not assert it
without a robust reproducibility guarantee. Setting
require_dominating=True still requests this stronger condition, and
LIMIT_REACHED remains an inconclusive search result.

```bash
uv run pytest packages/pycliques/tests/test_octahedral_compatibility_search.py -q
```

The 20 antipodal pairs are found during the test; none are encoded in
this algorithm. The historical original \(O_{20}\) was also found by
compatible-nonedge search, but its exact discovery trace was not saved.
Thus this test is a *new blind induced-octahedron discovery*, not a replay of
historical random choices; it does not certify complement domination or a retraction.

## Logical limitations

Complement domination (an empty intersection of the closed
neighborhoods of the \(2d\) octahedral vertices) is **necessary** for a
retraction onto \(O_d\), but **not sufficient**. The independent CSP
solver find_octahedral_retraction must find and certify a retraction
onto a particular embedded octahedron; if it returns EXHAUSTED then
that particular embedding is not a retract, and LIMIT_REACHED is
inconclusive.

This implementation addresses the from-scratch embedding discovery
stage. A fully blind search that also reliably identifies a
retracting \(O_{24}\) in the snub's fourth iterate remains to be
established.
