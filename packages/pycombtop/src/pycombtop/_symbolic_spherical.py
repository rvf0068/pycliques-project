"""Integral spherical-generation certificates on an existing Morse matching."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from itertools import islice, product

import networkx as nx
from pycliques.octahedral_nonlocal_search import (
    enumerate_induced_octahedra,
    verify_induced_octahedron,
)
from sympy import ZZ, Matrix, zeros
from sympy.matrices.normalforms import hermite_normal_form, smith_normal_form

from ._symbolic_morse_gradient import _MorseLimit, signed_projector

AntipodalPairs = tuple[tuple[Hashable, Hashable], ...]


@dataclass(frozen=True)
class SphericalCycle:
    """An induced octahedral sphere and its signed Morse projection."""

    pairs: AntipodalPairs
    projection: tuple[int, ...]


@dataclass(frozen=True)
class SphericalDegreeCertificate:
    """Selected cycles and integral lattice information in one degree."""

    degree: int
    betti: int
    cycles: tuple[SphericalCycle, ...]
    lattice_rank: int
    smith_factors: tuple[int, ...]
    complete: bool


@dataclass(frozen=True)
class SphericalGenerationCertificate:
    """Partial or complete spherical-generation proof.

    A complete proof covers every required higher degree. A limited or
    incomplete proof gives no negative statement about the homotopy type.
    Selected cycles can seed a later attempt.
    """

    degrees: tuple[SphericalDegreeCertificate, ...]
    complete: bool
    limited: bool
    states: int
    candidates: int


def _boundary(
    boundaries: dict[int, tuple[tuple[int, ...], ...]],
    faces: dict[int, list[int]],
    degree: int,
):
    rows = boundaries.get(degree)
    if rows is not None:
        return Matrix(rows)
    return zeros(len(faces.get(degree - 1, [])), len(faces.get(degree, [])))


def _project_sphere(
    pairs: AntipodalPairs,
    index: dict[Hashable, int],
    size: int,
    project: Callable[[int], tuple[int, ...]],
    charge: Callable[[], None],
) -> tuple[int, ...]:
    """Orient a join of ordered 0-spheres using the matching vertex order."""
    indexed = tuple((index[u], index[v]) for u, v in pairs)
    result = [0] * size
    for choices in product((0, 1), repeat=len(pairs)):
        charge()
        vertices = [pair[b] for pair, b in zip(indexed, choices, strict=True)]
        parity = sum(choices) + sum(
            u > v for i, u in enumerate(vertices) for v in vertices[i + 1 :]
        )
        sign = -1 if parity % 2 else 1
        value = project(sum(1 << u for u in vertices))
        for j, coefficient in enumerate(value):
            result[j] += sign * coefficient
    return tuple(result)


def spherical_generation(
    graph: nx.Graph,
    vertices: Sequence[Hashable],
    faces: dict[int, list[int]],
    boundaries: dict[int, tuple[tuple[int, ...], ...]],
    betti: dict[int, int],
    pivot: Callable[[int], int | None],
    max_states: int,
    *,
    seed: Iterable[AntipodalPairs] = (),
    search: bool = True,
) -> SphericalGenerationCertificate:
    """Try to saturate each higher integral homology lattice.

    The caller certifies simple connectivity and torsion-free homology.
    Hurewicz and relative Hurewicz handle the two lowest consecutive
    degrees. Other nonzero degrees need explicit spherical generators.
    """
    states = candidates = 0
    limited = False
    degrees: list[SphericalDegreeCertificate] = []
    index = {v: i for i, v in enumerate(vertices)}
    # Keep supplied seeds bounded even for arbitrary lazy iterables.
    supplied = tuple(islice(seed, max_states + 1))
    if len(supplied) > max_states:
        raise ValueError("too many spherical seeds for the state budget")
    if any(not verify_induced_octahedron(graph, pairs) for pairs in supplied):
        raise ValueError("spherical seeds must be induced octahedra")
    required = [d for d in sorted(betti) if d >= min(betti) + 2] if betti else []

    def charge(amount: int = 1) -> None:
        nonlocal states
        if states + amount > max_states:
            raise _MorseLimit
        states += amount

    for degree in required:
        lower = faces[degree]
        down = _boundary(boundaries, faces, degree)
        up = _boundary(boundaries, faces, degree + 1)
        lattice = hermite_normal_form(up)
        target_rank = lattice.cols + betti[degree]
        selected: list[SphericalCycle] = []
        pending: list[SphericalCycle] = []
        project = signed_projector(lower, pivot, charge)
        factors: tuple[int, ...] = ()
        seen: set[frozenset[frozenset[Hashable]]] = set()

        def flush() -> bool:
            nonlocal lattice, factors
            if pending:
                columns = Matrix([w.projection for w in pending]).T
                updated = hermite_normal_form(lattice.row_join(columns))
                if updated != lattice or not search:
                    selected.extend(pending)
                    lattice = updated
                pending.clear()
            if lattice.cols != target_rank:
                return False
            normal = smith_normal_form(lattice, domain=ZZ)
            factors = tuple(
                abs(int(normal[i, i]))
                for i in range(min(normal.shape))
                if normal[i, i]
            )
            return len(factors) == target_rank and all(x == 1 for x in factors)

        def add(pairs: AntipodalPairs) -> bool:
            nonlocal candidates
            charge()
            key = frozenset(frozenset(p) for p in pairs)
            if key in seen:
                return False
            seen.add(key)
            if not verify_induced_octahedron(graph, pairs):
                raise ArithmeticError("invalid enumerated octahedron")
            candidates += 1
            value = _project_sphere(pairs, index, len(lower), project, charge)
            if any(down * Matrix(value)):
                raise ArithmeticError("projected sphere is not a Morse cycle")
            pending.append(SphericalCycle(pairs, value))
            return len(pending) >= 16 and flush()

        complete = False
        try:
            for pairs in supplied:
                if len(pairs) == degree + 1 and add(pairs):
                    complete = True
                    if search:
                        break
            if not complete:
                complete = flush()
            if search and not complete:
                enumeration = enumerate_induced_octahedra(
                    graph,
                    target_dimension=degree + 1,
                    max_states=max(1, max_states - states),
                )
                previous = 0
                for pairs in enumeration:
                    charge(enumeration.states - previous)
                    previous = enumeration.states
                    if add(pairs):
                        complete = True
                        break
                charge(enumeration.states - previous)
                limited |= enumeration.limited
        except (_MorseLimit, RecursionError, MemoryError):
            limited = True
        finally:
            project.cache_clear()
        # At most one small already-projected batch remains. Retain it
        # even on search exhaustion so completed work is not lost.
        complete = flush() or complete
        if lattice.cols != target_rank:
            normal = smith_normal_form(lattice, domain=ZZ)
            factors = tuple(
                abs(int(normal[i, i]))
                for i in range(min(normal.shape))
                if normal[i, i]
            )
        degrees.append(
            SphericalDegreeCertificate(
                degree,
                betti[degree],
                tuple(selected),
                int(lattice.cols),
                factors,
                complete,
            )
        )
        if limited:
            break
    return SphericalGenerationCertificate(
        tuple(degrees),
        len(degrees) == len(required) and all(d.complete for d in degrees),
        limited,
        states,
        candidates,
    )
