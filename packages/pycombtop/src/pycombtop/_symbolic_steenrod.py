"""Evaluate Sq^2 over GF(2) on symbolic Morse homology generators.

The matching and its gradient pivot lookup belong to the existing Morse
engine. The cup-i formula uses the same ordered simplicial vertices.
Only the supports of lifted target cycles and requested source faces
are visited; the full simplicial complex is never enumerated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from ._symbolic_morse_gradient import _MorseLimit


@dataclass(frozen=True)
class SymbolicSq2Certificate:
    """Matrix of Sq^2 in dual Morse homology bases, packed by target rows."""

    degree: int
    source_rank: int
    target_rank: int
    rows: tuple[int, ...]
    rank: int
    lifted_faces: int
    states: int


def _bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _reduce(mask: int, basis: dict[int, int]) -> int:
    while mask:
        pivot = mask.bit_length() - 1
        if pivot not in basis:
            break
        mask ^= basis[pivot]
    return mask


def _columns(rows: tuple[tuple[int, ...], ...], n: int) -> list[int]:
    return [sum((row[j] & 1) << i for i, row in enumerate(rows)) for j in range(n)]


def _homology_basis(
    n: int, down: list[int], up: list[int]
) -> tuple[list[int], list[int]]:
    """Return cycles modulo boundaries and the dual cocycles."""
    span: dict[int, int] = {}
    for x in up:
        reduced = _reduce(x, span)
        if reduced:
            span[reduced.bit_length() - 1] = reduced
    boundary_rank = len(span)
    pairs: dict[int, tuple[int, int]] = {}
    cycles = []
    homology_pivots: dict[int, int] = {}
    for j, x in enumerate(down):
        v = 1 << j
        while x:
            p = x.bit_length() - 1
            if p not in pairs:
                pairs[p] = x, v
                break
            b, c = pairs[p]
            x ^= b
            v ^= c
        if x:
            continue
        v = _reduce(v, span)
        if v:
            p = v.bit_length() - 1
            span[p] = v
            homology_pivots[p] = len(cycles)
            cycles.append(v)
    if len(cycles) != n - len(pairs) - boundary_rank:
        raise ArithmeticError("inconsistent Morse homology basis")
    cocycles = []
    for j in range(len(cycles)):
        a = 0
        for p, x in sorted(span.items()):
            if (x & a).bit_count() % 2 != (homology_pivots.get(p, -1) == j):
                a |= 1 << p
        if any((a & b).bit_count() % 2 for b in up):
            raise ArithmeticError("Morse cochain is not a cocycle")
        if any((a & c).bit_count() % 2 != (i == j) for i, c in enumerate(cycles)):
            raise ArithmeticError("Morse bases are not dual")
        cocycles.append(a)
    return cycles, cocycles


def compute_sq2(
    faces: dict[int, list[int]],
    boundaries: dict[int, tuple[tuple[int, ...], ...]],
    pivot: Callable[[int], int | None],
    degree: int,
    max_states: int,
) -> SymbolicSq2Certificate:
    """Compute the square using batched mod-2 projections and cycle lifts."""
    states = 0

    def charge() -> None:
        nonlocal states
        states += 1
        if states > max_states:
            raise _MorseLimit

    def bases(k: int) -> tuple[list[int], list[int]]:
        return _homology_basis(
            len(faces.get(k, [])),
            _columns(boundaries.get(k, ()), len(faces.get(k, []))),
            _columns(boundaries.get(k + 1, ()), len(faces.get(k + 1, []))),
        )

    _, co = bases(degree)
    cy, _ = bases(degree + 2)
    if not co or not cy:
        return SymbolicSq2Certificate(
            degree, len(co), len(cy), (0,) * len(cy), 0, 0, states
        )

    def annotations(masks: list[int], vectors: list[int]) -> dict[int, int]:
        return {
            m: sum(((a >> j) & 1) << i for i, a in enumerate(vectors))
            for j, m in enumerate(masks)
        }

    ann = annotations(faces.get(degree, []), co)
    active: set[int] = set()

    @lru_cache(maxsize=65_536)
    def project(m: int) -> int:
        charge()
        if m in ann:
            return ann[m]
        v = pivot(m)
        if v is None or m in active:
            raise ArithmeticError("invalid Morse projection")
        if m >> v & 1:
            return 0
        active.add(m)
        try:
            value = 0
            t = m | (1 << v)
            for b in _bits(m):
                value ^= project(t ^ (1 << b))
            return value
        finally:
            active.remove(m)

    # Reverse the gradient projection: lift target cycles along upward
    # matching paths. The graph is acyclic by the Dong matching theorem.
    adj: dict[int, list[int]] = {}
    post: list[int] = []
    visiting: set[int] = set()

    def visit(m: int) -> None:
        if m in adj:
            return
        charge()
        if m in visiting:
            raise ArithmeticError("cyclic Morse lift")
        visiting.add(m)
        targets = []
        for v in _bits(m):
            charge()
            f = m ^ (1 << v)
            p = pivot(f)
            if p is not None and not (f >> p & 1):
                t = f | (1 << p)
                if t != m:
                    targets.append(t)
                    visit(t)
        visiting.remove(m)
        adj[m] = targets
        post.append(m)

    initial = annotations(faces.get(degree + 2, []), cy)
    weights = dict(initial)
    for m, a in initial.items():
        if a:
            visit(m)
    for m in reversed(post):
        w = weights.get(m, 0)
        if w:
            for t in adj[m]:
                charge()
                weights[t] = weights.get(t, 0) ^ w
    lifted = {m: w for m, w in weights.items() if w}

    # Verify cycle boundary and projection onto the prescribed critical
    # coefficients independently of the lifting recurrence.
    bd: dict[int, int] = defaultdict(int)
    for m, w in lifted.items():
        charge()
        p = pivot(m)
        if p is not None and not (m >> p & 1):
            raise ArithmeticError("lift contains an upward-matched face")
        for v in _bits(m):
            charge()
            bd[m ^ (1 << v)] ^= w
    if any(bd.values()) or any(
        lifted.get(m, 0) != w for m, w in initial.items()
    ):
        raise ArithmeticError("invalid lifted Morse cycles")

    # Medina-Mardones cup-(degree-2) formula: delete four vertices
    # partitioned by the parity of their positions in the ordered tuple.
    patterns = []
    for u in combinations(range(degree + 3), 4):
        charge()
        pos = tuple(x for j, x in enumerate(u, 1) if x % 2 == j % 2)
        neg = tuple(x for j, x in enumerate(u, 1) if x % 2 != j % 2)
        if len(pos) == len(neg) == 2:
            patterns.append((pos, neg))
    rows = [0] * len(cy)
    try:
        for m, w in lifted.items():
            vertices = [1 << i for i in _bits(m)]
            value = 0
            for (a, b), (c, e) in patterns:
                charge()
                x = project(m ^ vertices[a] ^ vertices[b])
                if x:
                    value ^= x & project(m ^ vertices[c] ^ vertices[e])
            for i in _bits(w):
                rows[i] ^= value
        basis: dict[int, int] = {}
        for row in rows:
            reduced = _reduce(row, basis)
            if reduced:
                basis[reduced.bit_length() - 1] = reduced
        return SymbolicSq2Certificate(
            degree, len(co), len(cy), tuple(rows), len(basis), len(lifted), states
        )
    finally:
        project.cache_clear()
