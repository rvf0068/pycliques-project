# Certified snub-disphenoid divergence: an octahedral retraction

The snub disphenoid graph \(S\) (graph6 `GQyuzw`) satisfies

\[
K^4(S)\longrightarrow O_{24}
\]

by an explicit graph retraction. The source has 1,076 vertices and 504,360
edges; the target is the complement of 24 disjoint edges on 48 vertices.
Neumann-Lara's retraction theorem therefore proves that \(S\) is
\(K\)-divergent.

## Normal CI: independently replay the certificate

Run from the monorepo root:

```bash
uv run pytest packages/pycliques/tests/test_octahedral_retractions.py -m 'not slow'
```

The test reconstructs the first four clique iterates and verifies the
stored 1,076-image map using
`pycliques.verify_retraction_witness`, **without invoking the search**.
The fixture in `packages/pycliques/tests/data/snub_o24_retraction.json.b85`
is zlib-compressed JSON encoded using Python's Base85. Its JSON object
has fields `target_pairs` (24 ordered antipodal pairs in source labels)
and `images` (the image of each source vertex in integer-label order).

The certificate uses the canonical \(F_{4,4}\) vertex numbering and
labels each clique iterate by increasing vertex bitmask. The graph6
labeling of `snub_disphenoid()` is isomorphic but not identical. These
canonical labels are reconstructed explicitly in the test.

## Optional rediscovery, with the embedding fixed

```bash
uv run pytest packages/pycliques/tests/test_octahedral_retractions.py -m slow
```

`pycliques.find_octahedral_retraction(G, antipodal_pairs)` searches
for a retraction fixing a *given induced octahedron*. The method uses
bitset image domains, singleton propagation of forbidden antipodal
images, and minimum-remaining-values backtracking. Its deterministic
seed allows replay. A returned witness is checked with
`verify_retraction_witness`.

The status is one of `FOUND`, `EXHAUSTED`, or `LIMIT_REACHED`.
`EXHAUSTED` excludes a retraction onto **that particular embedding**
only; `LIMIT_REACHED` is inconclusive. The solver does not discover an
induced octahedron automatically and should not be confused with the
generic `retracts(G, H)` search, which also searches over embeddings.

The fast certificate test belongs to normal CI; the rediscovery test
is marked `slow`, in accordance with the repository's pytest settings.

## Fresh search through classify_clique_behavior

This opt-in classifier mode is available from the Python API:

```python
import networkx as nx
from pycliques.small import classify_clique_behavior

G = nx.from_graph6_bytes(b"GQyuzw")
result = classify_clique_behavior(
    G,
    tries=5,
    bound=1200,
    clockwork_depth=0,
    retraction_depth=0,
    octahedron_retract_presets=("snub_o24",),
    octahedron_max_states=5000,
)
print(result.verdict)
print(result.reason)
print(result.certificate.rule if result.certificate else None)
```

The seeded mode uses the *induced O24 embedding*, transported from the
original labeled F(4,4) to the input labeling (including graph6) and then
through the nested clique labels. It searches for a **new** retraction: the
previously published projection/images are not loaded. Since all five
snub iterates through K^4 are stiff, paring leaves the target intact.

The preset requires `tries >= 5`, `bound >= 1076`. The specialized
search respects `octahedron_max_states`; the generic
`octahedron_max_patterns` and `octahedron_max_assignments` settings
do not apply to it. A bound hit is inconclusive, not proof that a
retraction does not exist.

Without the preset, the snub is now immediately recognized as a
**proven** divergent reference (following the certified retraction in
PR #27), and the expensive fourth-iterate search is not repeated.

The existing unseeded options, for example
`octahedron_retract_iterates=(3,)` and `octahedron_dimensions=(4,)`,
continue to search for *unknown* octahedral embeddings. These should
not be used to enumerate arbitrary O24 embeddings in the enormous K4
snub graph.

## Literature

F. Larrión, M. A. Pizaña, and R. Villarroel-Flores,
*Contractibility and the clique graph operator*, Discrete Mathematics
**308** (2008), 3461–3469, Theorem 2.4 (retraction principle).
