# Implicit octahedral amplification and complete transversals

An induced octahedron of dimension d is specified by its d antipodal
pairs. Suppose that no vertex of the host graph is equal or adjacent
to every octahedral vertex. Every choice of one endpoint of each pair
is complete and extends to a maximal clique. Nonopposite choices
intersect, while opposite extensions are disjoint: a vertex common
to opposite extensions would be a common closed neighbor of the seed.
Thus the 2^d extensions induce an octahedron of dimension 2^(d-1)
in the clique graph.

The pre-existing amplify_induced_octahedron API now delegates to
certify_octahedral_amplification, which additionally returns every
chosen extension and the exact common closed neighborhood C_s of
each seed transversal s. The latter is the union of all maximal
cliques extending s. The certificate can be checked without
constructing the host clique graph or comparing all amplified pairs.

```python
from pycliques import (
    RetractionSearchStatus,
    amplified_common_neighbor,
    certify_octahedral_amplification,
    octahedron,
    verify_octahedral_amplification,
)

G = octahedron(3)
cert = certify_octahedral_amplification(G, [(0, 1), (2, 3), (4, 5)])
assert cert.dimension == 4
assert verify_octahedral_amplification(G, cert)
result = amplified_common_neighbor(G, cert)
assert result.status is RetractionSearchStatus.EXHAUSTED
```

The general function find_complete_transversal(G, family) searches
for a nonempty complete set that meets every member of family.
A complete transversal exists precisely when a maximal clique
transversal exists. Branch on an unmet member, retaining compatible
vertices and removing earlier alternatives to partition the search.
A state with an unmet member disjoint from all available vertices
cannot succeed. FOUND carries a verified witness; EXHAUSTED means
there is no such set; LIMIT_REACHED is inconclusive.

For an amplification certificate, searching for a transversal of the
extensions tests for a common closed neighbor of the amplified
octahedron. With all_extensions=True, search instead over the sets C_s.
EXHAUSTED proves that no choice of maximal extensions could have
a common neighbor. FOUND in this stronger test does not by itself
give a common neighbor of the particular chosen extensions.

The function find_split_cover_counterexample searches for a complete
set meeting both sides of every supplied bipartition. EXHAUSTED
certifies that every maximal clique lies entirely in a side of at least
one bipartition. This verifies the coverage part of the split-cover
criterion, not containment of the prescribed octahedral endpoints.

## Limits of repeated amplification

Two successful amplifications do not guarantee a third along the
same chosen seed. The 11-vertex graph with graph6 J]~vJZZn}~? has
a complement-dominating O3 on pairs (0,1),(2,3),(4,5). Its clique
graph has 21 vertices. The first amplification to O4 has no common
closed neighbor, independently of extension choices, as the complete
transversal search over C_s proves. However, all eight first-stage
extension choices yield a second-stage O8 for which a common neighbor
is unavoidable, regardless of second-stage choices.

To check the last claim, let M_t be the intersection of all maximal
cliques extending a second-stage seed transversal t. Any complete
set meeting every M_t extends to a clique meeting every possible
selection of the second-stage maximal extensions. The regression
enumerates the small clique graph, the eight first-stage choices,
and verifies the existence of these forced transversals.

This example does not classify the graph's clique behavior or rule
out amplification starting from a different seed.

## Cubic graph 65

The slow regression rebuilds the 140-vertex third pared iterate of the
complement of the cubic graph KsX_o_C?_B_M using stable integer clique
IDs, then replays a known O14 seed. It certifies the induced O8192
and its absence of a common neighbor for every choice of extensions.
Hence the fifth ordinary clique iterate has at least 2^8192 vertices.
This is finite growth, not a proof of divergence.

```sh
uv run pytest -m slow packages/pycliques/tests/test_implicit_amplification.py
```
