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
* Computation uses exact symbolic linear algebra (SymPy), so it is intended
  for small complexes.
