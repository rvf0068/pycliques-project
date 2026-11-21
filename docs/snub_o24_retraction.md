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

## Literature

F. Larrión, M. A. Pizaña, and R. Villarroel-Flores,
*Contractibility and the clique graph operator*, Discrete Mathematics
**308** (2008), 3461–3469, Theorem 2.4 (retraction principle).
