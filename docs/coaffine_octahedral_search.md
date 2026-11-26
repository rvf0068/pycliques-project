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

## Optional SAT backend (Linux)

The same public function also supports `backend="sat"`. Install the optional
system Z3 shared library, e.g. `sudo apt install libz3-4` on Ubuntu. No Python
dependency, compilation, or change to the default backend is required.

```python
result = pycliques.find_octahedral_retraction_coaffine(
    P,
    backend="sat",
    min_dimension=3,
    max_states=100_000,
    time_limit=60,
    sat_max_clauses=4_000_000,
    sat_memory_limit_mb=2048,
)
```

For this backend only, `max_states` is a Z3 conflict budget and
`states_explored` reports solver conflicts (zero if unavailable). The other
CSP counters are zero. `value_order="fiber"` is unsupported. Z3 chooses its
own branching policy. This is an explicit opt-in; auto never chooses SAT.

The compressed encoding uses assignment variables x(u,v), image variables
y(v), and antipodal-pair variables p(v,w). For each source u exactly one
x(u,v) holds, and v must be a nonneighbor distinct from u. Assignment
implies image membership, membership implies some incident pair, and
p(v,w) implies reciprocal assignments and both memberships. If two
selected vertices are nonadjacent they must be paired. Consequently the
complement of the selected image is a perfect matching. The edge-image
constraints forbid mapping adjacent sources to distinct nonadjacent
targets. A threshold counter requires at least 2*min_dimension selected
vertices. These conditions are equivalent to the existence of a coaffine
endomorphism f with f³=f whose image is an octahedron of the specified
minimum dimension. Its square is the desired retraction.

Balanced OR trees share subsets of inverse domains in the edge-image
constraints. Each internal literal is equivalent to its children's
disjunction. This is an exact existential CNF compression, not a heuristic.

DIMACS is streamed to a temporary file, with both its variable and clause
counts bounded by `sat_max_clauses`. The isolated Linux solver subprocess
has an address-space limit (`sat_memory_limit_mb`, in MiB); the limit does
not bound memory used by the caller's graph or the Python encoder. A finite
`time_limit` includes construction, loading and solving. Z3 UNKNOWN, a
worker failure, or an encoding/resource limit yields LIMIT_REACHED, never
EXHAUSTED. Only an exact UNSAT response or an elementary impossibility
precheck yields EXHAUSTED. The existing Python certificate checks verify
every returned map and retraction independently. A missing optional
system library raises RuntimeError.

## Incidence-based octahedral retractions

For a clique iterate, retain its clique supports using `ImplicitCliqueGraph`.
The new specialized entry point shares the ordinary retraction result and
verification conventions, but operates on the intersection representation.

```python
import pycliques as pc

# base is the preceding pared iterate, not the host itself.
raw = pc.ImplicitCliqueGraph.from_graph(base, max_cliques=3000)
if raw is not None:
    pared = raw.pare()
    assert pc.verify_implicit_paring(raw, pared)
    proof = pc.find_octahedral_retraction_incidence(
        pared.core, backend="hybrid", max_steps=100_000,
        time_limit=60, max_states=500_000,
    )
```

Compile the optional weighted walk explicitly with
`python -m pycliques.octahedral_incidence_native --build` (GNU C++17/libstdc++).
The walk currently supports at most 64 ground atoms and 1280 active host
vertices. Its state is a family of compatible antipodal pairs and cuts
separating their supports. A cut covers every support contained wholly in
one shore. An anchored weighted minimum-cut optimization favors uncovered
and singly covered supports; adaptive pair replacement and random restarts
repair holes without fixing the target dimension in advance. A zero-miss
cover is an exact certificate. Every native checkpoint is checked, including
when a timeout occurs. Checkpoints use original active clique IDs and masks
in the host's base-vertex order. `states_explored` records the best checkpoint
step (not the total number of walk moves).

The exact `backend="sat"` works with wider support masks and uses the same
optional Linux Z3 worker and resource limits as the general coaffine backend.
For a choice f(u)=v, all atoms of the support of u must belong to the
fibre union for v. No atom may belong to two opposite fibre unions. One
true image choice per source suffices; the selected endpoints must induce
a matching in the complement. This is equivalent to the existence of a
retraction, by f = antipodal involution composed with the retraction.

`retained_pairs` optionally restricts exact completion. EXHAUSTED then means
no solution containing those pairs, not global nonexistence. In hybrid mode
the walk uses half the wall budget and SAT uses the remaining time to attempt
completion retaining a deterministic random half of the best pairs.
An unsuccessful hybrid search is inconclusive. The optional
`starting_matchings` are restart candidates only. No native executable
is built implicitly. Positive maps are independently verified after discovery;
this final verification is not included in the discovery time budget.

The result adds `partial_cuts`, `uncovered`, and `retained_pairs`.
If SAT completes a partial cover, the partial-cut fields still describe the
walk checkpoint, not necessarily the returned retraction. Use
`antipodal_pairs` and `retraction` for the final witness.
