# General retraction obstructions and implicit clique graphs

These opt-in tools complement (rather than change the defaults of) the
ordinary clique-behavior classifier. They were motivated by the two
remaining cubic complements of order twelve.

## Coaffine weak endomorphisms

`find_coaffine_endomorphism(G, max_states=100000, time_limit=None)`
searches for a weak endomorphism f such that f(v) is neither v nor
adjacent to v, for every vertex v. Unlike a coaffination (automorphism),
f need not be bijective. The search uses bitmask domains, arc consistency,
and minimum-remaining-values branching, with no assumption on f's square.
`verify_coaffine_endomorphism(G, f)` checks a positive certificate.

If r:G -> H is a retraction, i is inclusion, and f is coaffine on H,
then F = i f r is a coaffine endomorphism of G. Otherwise F(x) would be
equal or adjacent to x; applying r contradicts coaffinity of f at r(x).
Consequently EXHAUSTED excludes retractions onto **every** target
admitting a coaffine endomorphism, not just octahedra, including cycle
complements and suspensions of cycles of length at least four.
It is not a convergence test. LIMIT_REACHED gives no obstruction.

Both coaffine searches now share the optional native backend and the
portable transposed-domain search. Use backend='native', value_order='fiber'
for the fiber-first search, after explicitly building the native engine.
The octahedral search accepts only leaves whose squares are octahedral
retractions; the general search accepts any verified coaffine map.

## Proper nontrivial stiff retracts

`search_proper_stiff_retraction(G, ...)` enumerates induced subsets of
orders from min_target_order (default 2) through max_target_order
(default |G|-1, always at most |G|-1). Stiff means no distinct vertices
have nested closed neighborhoods. Every external vertex must have an
admissible image compatible with its fixed target neighbors. After
these exact necessary tests, the existing bounded retraction solver
searches the remaining candidates. Positive witnesses are independently
verified. EXHAUSTED applies only to the requested order interval.

The subset, state and assignment budgets are global across candidates.
Every visited subset counts, even one discarded by a cheap filter.
The default cap is intentionally much smaller than a complete scan of
a 24-vertex host. Exhausting a budget is inconclusive. A proper divergent
retract can be pared to a nontrivial stiff divergent retract, so an
exhaustive negative result over the whole interval excludes all proper
divergent retracts of the host.

## Implicit clique graphs

```python
import networkx as nx
from pycliques import ImplicitCliqueGraph, verify_implicit_paring

X = nx.cycle_graph(6)
Y = ImplicitCliqueGraph.from_graph(X, max_cliques=100000)
assert Y is not None
result = Y.pare()
assert verify_implicit_paring(Y, result)
P = result.core
print(len(P), P.number_of_edges())
# Explicit conversion is optional, and can be guarded by an edge cap.
small = P.to_networkx(max_edges=100000)
```

Each maximal clique is encoded by an arbitrary-precision Python integer
mask over base vertices. Stable integer clique IDs are positions in the
sorted tuple of masks. The incidence mask for a base vertex records all
cliques containing it. An active mask records the induced subgraph.
`clique(id)` decodes the original labels. `closed_neighborhood_mask(id)`
is a mask of active clique IDs, not of base vertices.
No edge list is materialized unless to_networkx is called. The explicit
conversion retains integer clique IDs, and returns None before building
edges if the edge cap would be exceeded.

For an active family F, put D(q) = union of r in F disjoint from q.
Then q dominates p iff p is disjoint from D(q): every clique disjoint
from q is disjoint from p exactly when N[p] is contained in N[q].
The search computes D(q) implicitly with incidence masks, deletes all
dominated vertices other than q, and repeats until stiff. Each deletion
is recorded as (removed ID, dominator ID); deleting other vertices
preserves domination. The verifier checks the recorded moves using
closed-neighborhood containment and confirms final stiffness. The core
preserves both clique behavior and flag-complex homotopy type.

For the order-12 cubic graph encoded by `KsX@?oS@_D?J`, the complement's
second pared iterate has 64 vertices. Its clique graph has 80,909
vertices, and its stiff core has 76,787 vertices, with 2,896,524,461
edges. The slow regression reconstructs the graph by its graph6 code,
checks all 4,122 folds, and never materializes those edges. It can be
run explicitly with:

```sh
uv run pytest -m slow packages/pycliques/tests/test_implicit_clique_graph.py
```

## Mixed Morse dimensions in pycombtop

The existing facet and integral Morse certificates handle consecutive
positive critical dimensions d,d+1 with d>=2 using zero boundary and
Hurewicz. This now also covers d=1 when there is exactly one critical
1-cell. With one critical 0-cell the 1-skeleton is S^1. Each 2-cell
attaches by a map S^1 -> S^1; a zero integral degree is equivalent to
nullhomotopy. The rational top Betti test in facet Morse is sufficient:
maximal top Betti rank forces the integer boundary to be zero. The
symbolic integral implementation directly checks that d2 has rank zero
in the augmented one-circle case. This is separate from its existing
criterion in which d2 kills the circle.

For several critical 1-cells, a zero cellular boundary need not imply
nullhomotopy (e.g. the torus commutator attachment), so no extension to
that case is made. No wedge verdict is inferred from homology alone.
The regression cubic codes `I}KGGGB?w` and `IsX___J@o` are complements
with homotopy type S^1 wedge S^2.

References: Forman, Morse theory for cell complexes (1998);
Hatcher, Algebraic Topology (2002), Sections 1.1 and 4.2;
Dong, Topology of bounded-degree graph complexes (2003).
