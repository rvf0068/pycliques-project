# Homological filters for retractions

A graph retraction induces split injections in reduced homology, and clique
graphs preserve retractions. Thus a retraction from X onto T implies
beta_d(K^j(T)) <= beta_d(K^j(X)) for all degrees d and iterates j.

## Face-free upper bounds

For H the complement of X, choose u and delete its neighbors v_1,...,v_r.
Write H_i for the induced graph after deleting v_1,...,v_(i-1).
The deletion-link exact sequence for independence complexes yields

    beta_d(Ind(H_i)) <= beta_d(Ind(H_i-v_i))
                        + beta_(d-1)(Ind(H_i-N[v_i])).

The last deletion is a cone with apex u, so its reduced homology vanishes.
The sum of the link bounds is an upper bound for the original rank. The
empty augmented complex contributes beta_-1=1. Recursion stops at that
degree, at an isolated complement vertex (cone), or above the size bound.
The argument works over any field. A zero bound in a given degree also
proves integral vanishing by the universal coefficient theorem.

```python
from pycombtop import recursive_betti_upper_bounds
from pycliques import search_targeted_retractions

bounds = recursive_betti_upper_bounds(G, max_degree=5, max_states=100_000)
results = search_targeted_retractions(G, betti_upper_bounds=bounds.upper_bounds)
```

Positive bounds need not be exact, and the function does not certify
homotopy type. Only completed degrees appear, even when a state or recursion
limit interrupts the search. The bounded cache and search-state limit are
independent. The portable implementation does not promise the speed of the
earlier C++ research prototype.

Existing exact metadata from bounded_reduced_betti_mod2 remains supported.
If both exact ranks and upper bounds are supplied, exact ranks take precedence
and an inconsistent upper bound is rejected.

## Arbitrary reference graphs

For a target already known to be divergent, even a contractible one, compare
homology profiles relative to the starting graph:

```python
from pycliques import find_homological_retraction_obstruction

host = {0: {5: 0}, 1: {5: 0}, 2: {20: 0}}
target = {0: {5: 0}, 2: {20: 1}}
obstruction = find_homological_retraction_obstruction(host, target)
assert obstruction.iterate == 2 and obstruction.degree == 20
```

Host entries are certified upper bounds, target entries certified lower
bounds, all over GF(2). Exact ranks can be used on either side. Omitted
entries are unknown, not zero. The same relative iterate must be compared.
Pared iterates may be used since complete paring and clique iteration
preserve the relevant homotopy types up to strong equivalence.
A returned witness records the violated inequality; None does not establish
existence of a retraction. The routine deliberately does not compute clique
iterates or homology itself, and pycliques remains independent of pycombtop.

These obstructions also apply to special octahedra. If one transversal
{a_i} of an induced octahedron is a maximal complete set in the host,
fix the octahedron pointwise and send each external vertex to an antipode
b_i of a nonneighbor a_i. Such an i exists by maximality. No host edge
can map to an antipodal pair, so this is an ordinary graph retraction.

## Face-free exact homotopy certificates

Call `recursive_betti_upper_bounds(G)` (or pass `max_degree=None`) to
compute all degrees in one cached polynomial recurrence. The result contains
explicit bounds for degrees 0 through `len(G)-1`; a resource-limited run
returns no all-degree bounds. Passing an integer `max_degree` preserves
the degree-by-degree search and its completed-degree results.

The signed alternating sum of the polynomial coefficients is the exact
reduced Euler characteristic: each deletion-link step is an equality for
Euler characteristic. The `euler_refined_upper_bounds` property tightens
the field-independent bounds using this identity. Positive bounds alone
are not Betti ranks. The optional `neighbor_order` values `descending`,
`ascending`, and `reverse_ties` change the deletion order, not soundness.

```python
import networkx as nx
from pycombtop import (
    connected_link_order,
    recursive_betti_upper_bounds,
    verify_connected_link_order,
)

G = nx.octahedral_graph()
bounds = recursive_betti_upper_bounds(G)
assert bounds.reduced_euler_characteristic == 1
assert bounds.upper_bounds[2] == 1
order = connected_link_order(G)
assert order is not None and verify_connected_link_order(G, order)
```

A connected-link order starts at one vertex and adds vertices whose links
in the preceding induced graph are nonempty and connected. Van Kampen
then proves simple connectivity; failure of the search is inconclusive.

On graphs of order 20--256, the existing `homotopy_type_with_verdict`
cascade tries these bounded computations before symbolic Morse matching.
If every possible reduced homology degree but one is excluded over all
fields, simple connectivity and the universal coefficient theorem give
free integral homology in that one degree. Its rank is determined by
Euler characteristic. Hurewicz and Whitehead then prove the wedge of
spheres (or contractibility in the acyclic case). Mixed support is not a
wedge certificate. Uniform conicity is an alternative existing proof of
simple connectivity; no optimization or simplex enumeration is required.
Use `recursive_betti_states=0` to skip this strategy for the current call.
The total state budget is shared by the three fixed neighbor orders and
an adaptive fallback. All other
classification strategies and the vertex limit remain in effect.

### Adaptive local bounds

Pass `adaptive_pivots=2` to `recursive_betti_upper_bounds` in all-degree
mode to try two low-degree pivots and alternative neighbor orders at each
subproblem. Coefficientwise minima of valid bounds remain valid over every
field. Euler characteristics are carried separately: after taking minima
the alternating sum of bounds need not be the true Euler characteristic.
Disconnected complements split as joins, so the shifted bound polynomials
multiply. The state and cache budgets still apply; a limited run is
inconclusive. Fixed-policy and finite-degree calls retain their semantics.

### Integral homology in consecutive dimensions

The existing `symbolic_integral_morse_certificate` now also uses a
connected-link order when the integral Morse boundary computation has
torsion-free reduced homology supported in at most two consecutive degrees
at least two (or no reduced homology). The new `connectivity_order` field
records that sufficient witness when this fallback proves a wedge.

The theorem follows by choosing a wedge of lowest-dimensional spheres
representing an integral homology basis and taking the mapping cylinder
of its map into the simply connected complex. The pair is d-connected,
so relative Hurewicz and the exact sequences show that every homology
class in degree d+1 is spherical as well. The resulting integral homology
equivalence between simply connected CW complexes is a homotopy
equivalence. A gap of two or more degrees or nonzero torsion is not
covered by this criterion.

The classification cascade tries the natural symbolic order first, then
two distinct deterministic degree orders as bounded fallbacks. The
canonical-order Hopf verification is unchanged. Individual symbolic
certificate calls still accept an explicit vertex order for replay.

### Exact clique counts without a clique family

```python
import networkx as nx
from pycliques import count_maximal_cliques

G = nx.complete_multipartite_graph(*([2] * 24))
assert count_maximal_cliques(G) == 2**24
assert count_maximal_cliques(G.subgraph(range(20))) == 2**10
```

This new counting operation in the existing clique module does not allocate
the vertices or edges of the next clique graph. It counts independent
dominating sets of the complement by memoized branching. The state tracks
undominated vertices and eligible choices; each solution occurs in the
branch of its first chosen closed neighbor of the pivot. Complement
components of the remaining undominated vertices have independent
choices and multiply their counts. A resource limit returns None, not zero.

Every maximal clique of an induced subgraph extends to a distinct maximal
clique of the original graph, giving a certified lower bound on the order
of its clique graph. This is not by itself a clique-divergence certificate.
