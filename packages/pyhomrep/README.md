# pyhomrep

Group representations on the homology of simplicial complexes.

## Tutorial: decomposing H_r(K; ℂ) into irreducible S_n-modules

Let K be a simplicial complex on the vertices `0, 1, ..., n-1` that is
invariant under the symmetric group S_n permuting the vertices. Then S_n acts
on each homology group H_r(K; ℂ), and we want its decomposition into
irreducibles (Specht modules).

### 1. Define the complex with `pycombtop`

A `SimplicialComplex` is built from a base set plus either a list of facets or
a membership predicate, which must be invariant under vertex permutations.

```python
from pycombtop import SimplicialComplex

n = 4
# All subsets of {0,1,2,3} with at most 2 elements: K_4 as a 1-dimensional complex.
K = SimplicialComplex(range(n), function=lambda s: len(s) <= 2)
```

Other `pycombtop` constructors (`clique_complex`, `neighborhood_complex`,
`complex_of_forests`, ...) also work, as long as the vertices are the integers
`0, ..., n-1` and the construction is S_n-invariant. For instance,
`clique_complex(matching_graph(n))` is the matching complex of K_n.

### 2. Decompose the r-th homology

```python
from pyhomrep import decompose_p_homology

r = 1
mult = decompose_p_homology(K, n, r)
print({lam: m for lam, m in mult.items() if m != 0})
# {(2, 1, 1): 1}
```

The result is a dictionary indexed by partitions λ of n (as tuples);
`mult[λ]` is the multiplicity of the irreducible module S^λ in H_r(K; ℂ).
Here H_1(K_4) has rank 3 and is the irreducible S^(2,1,1).

To get all degrees at once, use `decompose_into_irreducibles(K, n)`, which
returns `{r: {λ: multiplicity}}` for `r = 0, ..., dim K`.

### 3. Sanity check

The character at the identity is the dimension of H_r, i.e. the Betti number:

```python
from sympy.combinatorics import Permutation
from pyhomrep import character_p_homology

print(character_p_homology(K, r, Permutation(n - 1)))  # 3
```

Another example, the matching complex of K_5:

```python
from pycombtop import clique_complex
from pyhomrep import decompose_p_homology, matching_graph

M = clique_complex(matching_graph(5))
print(decompose_p_homology(M, 5, 1))
# multiplicity 1 for (3, 1, 1), 0 for every other partition
```

### How it works

For each cycle type of S_n, the character of a representative on H_r is
`character_kernel(K, r, g) - character_image(K, r+1, g)` (trace on cycles minus
trace on boundaries). It is then expanded in the irreducible characters
(computed with the Murnaghan–Nakayama rule), weighting by conjugacy class sizes.

### Notes

* The vertices must be exactly `0, ..., n-1`; S_n acts by permuting them.
* The legacy character backend uses numerical SVD with integer rounding;
  the signed-orbit backend below uses exact rational ranks.


## Exact signed-orbit homology

For a finite simplicial group action, use `OrbitChainComplex` with explicit
generator maps on vertex labels. It computes rational coinvariant chains
using sparse integer boundaries and exact ranks. In characteristic zero,
coinvariant and invariant homology have the same dimension. Orientation-
reversing simplex stabilizers contribute no chain generator. Set
`reduced=True` for augmented chains and `relative=Y` for a relative complex.

For symmetric-group representations, `decompose_young_homology` uses the
Jacobi–Trudi formula and Young-subgroup invariants instead of dense character
matrices. For example:

```python
from pycombtop import clique_complex
from pyhomrep import bouc_decomposition, decompose_young_homology, matching_graph

n = 5
M = clique_complex(matching_graph(n))

def action(g, edge):
    return tuple(sorted(g(i) if i < g.size else i for i in edge))

assert decompose_young_homology(M, n, 1, action, reduced=True) == (
    bouc_decomposition(n, 1)
)
```

The same approach with `decompose_young_homology_map` computes the kernel,
image and cokernel of a simplicial map intertwining two explicit actions.
Only the requested skeleton is enumerated; this is not an implicit
orbit-generation algorithm and the trivial subgroup can still be expensive.
The existing SVD-based routines remain available. `YoungTableaux.CMNR`
now uses cached rim-hook recursion rather than enumerating tableaux.

For stabilizer calculations, `decompose_induced_character(n, weights, order)`
uses Frobenius reciprocity. The weights are sums of the subgroup character
over elements of each ambient symmetric-group cycle type.
