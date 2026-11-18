"""Exact signed gradient-path boundary recursion for a symbolic Dong matching."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from sympy import ZZ, Matrix
from sympy.matrices.normalforms import smith_normal_form


class _MorseLimit(Exception):
    """The gradient recursion exceeded a resource budget."""


def morse_boundaries(
    diagram, roots: Sequence[int], faces: dict[int, list[int]], max_states: int
):
    """Return (integral boundaries, ranks, torsion factors, visited states).

    No original simplices are enumerated. A face's matching pivot is the
    first stage at which it leaves the descending ZDD face family.
    """
    n = len(roots) - 1

    @lru_cache(maxsize=65_536)
    def member(stage: int, mask: int) -> bool:
        node = roots[stage]
        while node > 1:
            v = diagram.variables[node]
            if mask & ((1 << v) - 1):
                return False
            node = diagram.high[node] if mask & (1 << v) else diagram.low[node]
            mask &= ~(1 << v)
        return node == 1 and mask == 0

    @lru_cache(maxsize=65_536)
    def matched_pivot(mask: int) -> int | None:
        if member(n, mask):
            return None
        if not member(0, mask):
            raise ArithmeticError("gradient path left original complex")
        lo, hi = 0, n
        while lo + 1 < hi:
            middle = (lo + hi) // 2
            if member(middle, mask):
                lo = middle
            else:
                hi = middle
        return lo

    boundaries: dict[int, tuple[tuple[int, ...], ...]] = {}
    ranks: dict[int, int] = {}
    torsion: dict[int, tuple[int, ...]] = {}
    states = 0
    for degree in sorted(faces):
        if degree <= 0 or degree - 1 not in faces:
            continue
        lower, upper = faces[degree - 1], faces[degree]
        index = {face: i for i, face in enumerate(lower)}
        zero = (0,) * len(lower)
        active: set[int] = set()

        @lru_cache(maxsize=65_536)
        def project(face: int) -> tuple[int, ...]:
            nonlocal states
            states += 1
            if states > max_states:
                raise _MorseLimit
            if face in index:
                return tuple(int(i == index[face]) for i in range(len(lower)))
            if face in active:
                raise ArithmeticError("cyclic Morse gradient")
            active.add(face)
            try:
                pivot = matched_pivot(face)
                if pivot is None:
                    raise ArithmeticError("unexpected critical lower cell")
                if face & (1 << pivot):
                    return zero  # paired with a face of dimension one less
                coface = face | (1 << pivot)
                incidence = -1 if (coface & ((1 << pivot) - 1)).bit_count() % 2 else 1
                result = [0] * len(lower)
                remaining = coface & ~(1 << pivot)
                while remaining:
                    bit = remaining & -remaining
                    remaining ^= bit
                    sign = -1 if (coface & (bit - 1)).bit_count() % 2 else 1
                    value = project(coface ^ bit)
                    for i, c in enumerate(value):
                        result[i] -= incidence * sign * c
                return tuple(result)
            finally:
                active.remove(face)

        columns = []
        for face in upper:
            result = [0] * len(lower)
            remaining = face
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                sign = -1 if (face & (bit - 1)).bit_count() % 2 else 1
                value = project(face ^ bit)
                for i, c in enumerate(value):
                    result[i] += sign * c
            columns.append(tuple(result))
        project.cache_clear()
        rows = tuple(tuple(col[i] for col in columns) for i in range(len(lower)))
        boundaries[degree] = rows
        diag = smith_normal_form(Matrix(rows), domain=ZZ)
        factors = tuple(
            abs(int(diag[i, i]))
            for i in range(min(*diag.shape))
            if diag[i, i] != 0
        )
        ranks[degree] = len(factors)
        # This computes torsion of H_(degree-1), even if the next
        # boundary is nonzero: the exact sequence
        # 0 -> H_k -> coker(D_(k+1)) -> im(D_k) -> 0
        # has torsion-free quotient im(D_k) (a subgroup of free C_(k-1)).
        # Thus H_k and coker(D_(k+1)) have identical torsion subgroups.
        torsion[degree - 1] = tuple(x for x in factors if x > 1)
    return boundaries, ranks, {d: x for d, x in torsion.items() if x}, states
