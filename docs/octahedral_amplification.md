# Reusable induced-octahedron operations

The octahedral graph `O_d` is the complement of `d K_2`. Represent
an induced copy of `O_d` in a NetworkX graph by the list of its
`d` antipodal **nonedges**. Unlike the snub `snub_o24` preset, the
functions below do not contain any hardcoded witness coordinates.

## Three distinct tasks

```python
import pycliques

graph = pycliques.octahedron(4)

# Search for an arbitrary smaller induced octahedron.
seed = pycliques.find_induced_octahedron(graph, 3)
assert seed is not None

# One-for-two augmentation INSIDE graph: an induced O3 -> O4.
grown = pycliques.augment_induced_octahedron(
    graph, seed, max_walk_steps=0, require_dominating=True
)
assert grown is not None and len(grown) == 4

# Facet amplification IN THE NEXT CLIQUE GRAPH: O4 -> O8.
lift = pycliques.amplify_induced_octahedron(graph, grown)
assert len(lift) == 8
```

### Exact amplification theorem

If `pairs` induces `O_d` in `G` and the intersection of the
closed neighborhoods of all its `2d` vertices is empty (no
external common neighbor), take every binary transversal `q` of
its antipodal pairs and extend `q` greedily to an inclusion-maximal
complete set `M_q` of `G`. The `2^d` extended transversals are
distinct maximal completes, so they are vertices of `K(G)`.

Two `M_q` intersect when their transversals overlap; when they
are complementary, an intersection vertex would be adjacent to
all vertices of the original `O_d`, which is impossible.
Therefore they induce `O_(2^(d-1))` in `K(G)`.

`amplify_induced_octahedron` implements exactly this proof and
returns the antipodal pairs of maximal completes as
`frozenset` objects. **It does not enumerate the entire clique
graph**; it refuses to build more than `max_facets` extensions
(default 1024) and explicitly rejects octahedra with an external
common neighbor.

### Local one-for-two augmentation

`augment_induced_octahedron` takes an induced `O_d` and
iteratively considers replacing one antipodal pair with two pairs
compatible with all remaining vertices. If a direct augmentation
fails, it randomly moves among valid copies of `O_d`, using
`max_walk_steps`, `max_checks`, and a deterministic `seed`.
When `require_dominating=True` it only returns an `O_(d+1)`
that dominates in the complement. It is **heuristic**: returning
`None` never proves that no larger induced octahedron exists.

`find_induced_octahedron` uses a bounded induced-matching search
in the complement graph to find an initial `O_d`. It is suitable
for moderate dimensions but is not a practical unseeded search
for `O_24` among all 1076 vertices of `K^4(snub)`.

**How the snub certificate was actually discovered:** local
augmentation within the fourth iterate produced a chain
`O_20, O_21, O_22, O_23, O_24`; the separate exact amplification
theorem subsequently gives much larger induced octahedra in
later iterates. Neither operation is by itself a proof of divergence.
The decisive additional step was an independently verified
graph retraction to the induced `O_24`.

## Reproducibility

```bash
uv run pytest packages/pycliques/tests/test_octahedral_amplification.py
```

See also [the snub O24 retraction certificate](snub_o24_retraction.md)
and `pycliques.find_octahedral_retraction`, which searches for a
retraction onto a **specified** embedded octahedron.

For historical context, the special-facet/special-retraction
equivalence is Theorem 3.7 of Larrión--Pizaña--Villarroel-Flores,
*On the clique behavior of graphs with small constant link*,
Ars Combinatoria (2019). A special retraction is a sufficient
condition for divergence, but the snub certificate uses a general
retraction, not that special criterion.
