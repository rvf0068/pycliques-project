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
