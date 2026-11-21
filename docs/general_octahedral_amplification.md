# General induced-octahedron amplification

`find_induced_octahedron_amplified` searches an arbitrary finite simple
NetworkX graph for an induced octahedron `O_d = complement(d K_2)`. Unlike
the snub-specific seeded classifier, this routine does **not** contain
saved target vertices or a saved retraction map.

## Principle

Keep a random subset of the antipodal pairs of a current induced
octahedron. If the retained endpoints form a set `W`, another antipodal
pair `(u, v)` may be added whenever `u` and `v` are nonadjacent and
both belong to the intersection of the closed neighborhoods of the
vertices in `W`. This condition makes all four cross edges from the new
pair to each old pair automatic.

An iterative destroy-and-repair search can change most of the pairs;
it is **not** limited to extending the last target by one pair.
Bit masks represent the common closed-neighborhood intersection.

The option `require_dominating=True` demands an empty common
closed-neighborhood intersection of the final induced octahedron.
This condition supports the finite amplification theorem, but even a
dominating induced octahedron is **not**, in general, a retract.

## API

```python
import pycliques

result = pycliques.find_induced_octahedron_amplified(
    graph,
    target_dimension=24,
    seed_pairs=smaller_octahedron_pairs,  # optional: any valid induced O_d
    max_restarts=20_000,
    pair_samples=32,
    seed=24_000,
    require_dominating=True,
)

if result.status is pycliques.OctahedronSearchStatus.FOUND:
    pairs = result.antipodal_pairs
    assert pairs is not None
    assert pycliques.verify_induced_octahedron(
        graph, pairs, require_dominating=True
    )

    retraction = pycliques.find_octahedral_retraction(
        graph, pairs, max_states=10_000
    )
    # A theorem about divergence applies only if the status is FOUND and
    # the returned retraction witness passes independent verification.
```

The search budget is counted in randomized restarts; the retraction
solver has a **separate** `max_states` budget. A `LIMIT_REACHED`
result in either phase is inconclusive. Neither method proves that a
retraction does not exist outside its examined cases.

## Verified research benchmark

For the canonical bitmask-ordered fourth clique iterate of the
snub disphenoid, order 1,076 and size 504,360, the known 20-pair seed
from the independent F(m,n) exploration was used to **discover** a
different 24-pair induced octahedron, without any saved O24 pairs.

With search seed 24000, default sampling parameters, and at most
18,000 restarts, the procedure returned an induced complement-dominating
O24 after 7,596 restarts. An independent bounded CSP then found a
retraction onto that separately discovered target after 746 nodes.
All 504,360 host edges were checked for map preservation. The
20-pair starting coordinates are research input, **not embedded in
the general search implementation**.

The existing certified snub retraction in PR #27 is the permanent
mathematical proof. The present discovery experiment shows that the
method used in the search is available for other graphs, rather than
merely replaying the specific saved O24 witness.

For unusual graphs, begin with smaller targets or more restarts.
The heuristic can miss a target even when it exists; no negative
classification is inferred from an unsuccessful run.
