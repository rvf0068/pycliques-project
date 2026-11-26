"""Specht multiplicities from Young-subgroup homology invariants."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

from pycombtop import SimplicialComplex
from sympy.combinatorics import Permutation

from pyhomrep.orbits import OrbitChainComplex, homology_map_rank
from pyhomrep.utilities import partitions_list
from pyhomrep.young import YoungTableaux


def _partitions(n: int) -> list[tuple[int, ...]]:
    if n < 0:
        raise ValueError("n must be nonnegative")
    return [()] if n == 0 else [tuple(map(int, p)) for p in partitions_list(n)]


def _generators(alpha, action):
    start = 0
    result = []
    for size in alpha:
        for i in range(start, start + size - 1):
            g = Permutation(i, i + 1)
            result.append(lambda v, g=g: action(g, v))
        start += size
    return result


@cache
def _jacobi_trudi(lam: tuple[int, ...]):
    """Coefficients of h_alpha in s_lambda, grouping equal compositions."""
    length = len(lam)

    @cache
    def terms(i, mask):
        if i == length:
            return {(): 1}
        result: Counter = Counter()
        for j in range(length):
            a = lam[i] - i + j
            if mask & (1 << j) or a < 0:
                continue
            sign = (-1) ** (mask >> (j + 1)).bit_count()
            for tail, coeff in terms(i + 1, mask | (1 << j)).items():
                alpha = tuple(sorted((*tail, a), reverse=True)) if a else tail
                result[alpha] += sign * coeff
        return {a: c for a, c in result.items() if c}

    return terms(0, 0)


def specht_multiplicities(
    n: int, invariant_dimension: Callable[[tuple[int, ...]], int]
) -> dict[tuple[int, ...], int]:
    """Recover multiplicities from dimensions of Young-subgroup invariants."""
    value = cache(invariant_dimension)
    result = {
        lam: sum(c * value(a) for a, c in _jacobi_trudi(lam).items())
        for lam in _partitions(n)
    }
    if any(m < 0 for m in result.values()):
        raise ValueError("Inconsistent invariant dimensions")
    return {lam: m for lam, m in result.items() if m}


def decompose_young_homology(
    sc: SimplicialComplex,
    n: int,
    k: int,
    action: Callable[[Permutation, Any], Any],
    *,
    reduced: bool = False,
    relative: SimplicialComplex | None = None,
) -> dict[tuple[int, ...], int]:
    """Exact Specht decomposition via signed orbits, with explicit S_n action."""
    return specht_multiplicities(
        n,
        lambda a: OrbitChainComplex(
            sc, _generators(a, action), reduced=reduced, relative=relative
        ).betti_number(k),
    )


def decompose_young_homology_map(
    source: SimplicialComplex,
    target: SimplicialComplex,
    n: int,
    k: int,
    source_action: Callable[[Permutation, Any], Any],
    target_action: Callable[[Permutation, Any], Any],
    f: Callable[[Any], Any],
    *,
    reduced: bool = False,
) -> dict[str, dict[tuple[int, ...], int]]:
    """Specht multiplicities of kernel, image and cokernel of a simplicial map."""

    @cache
    def data(a):
        x = OrbitChainComplex(source, _generators(a, source_action), reduced=reduced)
        y = OrbitChainComplex(target, _generators(a, target_action), reduced=reduced)
        r = homology_map_rank(x, y, f, k)
        return x.betti_number(k) - r, r, y.betti_number(k) - r

    return {
        name: specht_multiplicities(n, lambda a: data(a)[i])
        for i, name in enumerate(("kernel", "image", "cokernel"))
    }


def bouc_decomposition(n: int, k: int) -> dict[tuple[int, ...], int]:
    """Bouc's reduced rational matching-complex homology formula."""
    result = {}
    for lam in _partitions(n):
        conjugate = tuple(
            sum(row >= j for row in lam) for j in range(1, max(lam, default=0) + 1)
        )
        diagonal = sum(row >= i for i, row in enumerate(lam, 1))
        if lam == conjugate and diagonal == n - 2 * (k + 1):
            result[lam] = 1
    return result


def decompose_induced_character(
    n: int, weighted_cycle_counts: Mapping[tuple[int, ...], int], subgroup_order: int
) -> dict[tuple[int, ...], int]:
    """Decompose an induced S_n-character using Frobenius reciprocity.

    Each weight is the sum of the subgroup character values on elements
    of the specified S_n cycle type. The input character must be integer
    valued. The caller supplies the subgroup order, including zero weights.
    """
    if subgroup_order <= 0:
        raise ValueError("Subgroup order must be positive")
    if any(
        sum(rho) != n or any(r <= 0 for r in rho) for rho in weighted_cycle_counts
    ):
        raise ValueError("Invalid cycle type")
    result = {}
    for lam in _partitions(n):
        value = sum(
            weight * YoungTableaux(list(lam), list(rho)).CMNR()
            for rho, weight in weighted_cycle_counts.items()
        )
        multiplicity, remainder = divmod(value, subgroup_order)
        if remainder or multiplicity < 0:
            raise ValueError("Weights do not give an induced character")
        if multiplicity:
            result[lam] = multiplicity
    return result
