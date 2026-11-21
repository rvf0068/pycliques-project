# Complete octahedral retraction search through coaffine endomorphisms

The function `find_octahedral_retraction_coaffine` finds **ordinary
graph retractions onto induced octahedra**, without knowing the
antipodal target pairs in advance. It uses a finite constraint
satisfaction search with deterministic bitmask arc consistency and
minimum-remaining-values branching.

**Mathematical completeness theorem.** Let `X` be a finite simple
graph and `O_d` an induced subgraph, `d >= 3`. If
`r : X -> O_d` is a graph retraction, let `i` denote the inclusion
and `tau` the antipodal involution on `O_d`. Define
`f = i ∘ tau ∘ r : X -> X`. This is a graph endomorphism under
the convention that an edge may map to an edge or equality. For
every vertex `v`, `f(v)` is neither equal nor adjacent to `v`:
otherwise the image of `v-f(v)` under `r` would be the
nonedge between `r(v)` and `tau(r(v))`. Furthermore
`f² = i ∘ r`, which is an idempotent map fixing the induced
`O_d` pointwise. Thus **every** octahedral retraction appears
in the exact search for coaffine endomorphisms, followed by testing
whether their squares are octahedral retractions.

Conversely, if a coaffine endomorphism `f` has a square `f²`
that is idempotent and whose image induces `O_d`, then `f²`
is a graph retraction onto this induced `O_d`. Every returned
`FOUND` projection is independently checked by
`verify_induced_octahedron` and `verify_retraction_witness`.

The search does **not** require `f` to be an automorphism.
It is therefore strictly more general than testing for coaffinations
(permutation automorphisms with no vertex sent to a neighbor).

## Public usage

```python
import networkx as nx
import pycliques
from pycliques.retractions import RetractionSearchStatus

G = nx.from_graph6_bytes(b"HCZTf\\}")  # Nine-vertex F_(4,5)
P = pycliques.completely_pared_graph(G)
for _ in range(4):
    P = pycliques.completely_pared_graph(pycliques.clique_graph(P))

result = pycliques.find_octahedral_retraction_coaffine(
    P, min_dimension=3, max_states=20000, time_limit=30
)
assert result.status is RetractionSearchStatus.FOUND
assert result.dimension == 9

target = P.subgraph(v for pair in result.antipodal_pairs for v in pair)
assert pycliques.verify_retraction_witness(
    P, target, (result.retraction, result.inclusion)
)
```

This **blind search** rediscovers the `O_9` retract of the
202-vertex fourth pared clique iterate of `F_(4,5)`,
without reading any stored target vertices or projections.
The full 202-entry independent certificate is archived in the
separate census research project.

`FOUND` proves the existence of a retraction; `EXHAUSTED`
certifies that no ordinary retraction onto any induced `O_d`,
`d >= min_dimension`, exists in the supplied graph; and
`LIMIT_REACHED` is strictly inconclusive. A bounded iterate
construction returning `None` is also inconclusive.

The algorithm does not itself classify the full clique behavior
of a graph: a proven retraction onto `O_d` with `d>=3`
does imply clique divergence by Neumann-Lara's retraction theorem,
while exhaustion of this search does **not** imply convergence.
