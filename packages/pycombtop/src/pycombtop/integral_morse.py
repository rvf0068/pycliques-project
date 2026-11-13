"""Elementary face-removal certificates with an integral Morse boundary.

A verified ordering removes a maximal critical face, or a free face and
its unique codimension-one coface. This certifies an acyclic matching.
With one critical vertex and remaining cells in dimensions d,d+1 (d>=2),
zero boundary in the integral Morse complex implies all attachments are
null by Hurewicz. See Forman, A User's Guide to Discrete Morse Theory,
Sections 2 and 7, and Hatcher, Algebraic Topology, Theorem 4.32.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from heapq import heappop, heappush

import networkx as nx

from .morse_certificates import _all_nonempty_faces, _bit_list


@dataclass(frozen=True)
class IntegralMorseCertificate:
    # Nonempty faces are bitmasks in the supplied complex. (0, high)
    # designates a critical maximal face, (low, high) a free pair.
    removals: tuple[tuple[int, int], ...]
    spheres: dict[int, int]


def _face_removals(faces: set[int], vertices: int) -> tuple[tuple[int, int], ...]:
    cofaces = dict.fromkeys(faces, 0)
    for face in faces:
        for bit in _bit_list(face):
            if lower := face ^ bit:
                cofaces[lower] += 1
    free: list[tuple[int, int]] = []
    maximal: list[tuple[int, int]] = []

    def queue(face: int) -> None:
        if cofaces[face] == 1:
            heappush(free, (-face.bit_count(), face))
        elif cofaces[face] == 0:
            heappush(maximal, (-face.bit_count(), face))

    def remove(face: int) -> None:
        del cofaces[face]
        for bit in _bit_list(face):
            if lower := face ^ bit:
                cofaces[lower] -= 1
                queue(lower)

    for face in faces:
        queue(face)
    steps = []
    while cofaces:
        while free and cofaces.get(free[0][1]) != 1:
            heappop(free)
        if free:
            _, low = heappop(free)
            high = next(
                low | b for b in _bit_list(vertices ^ low) if low | b in cofaces
            )
        else:
            while cofaces.get(maximal[0][1]) != 0:
                heappop(maximal)
            _, high = heappop(maximal)
            low = 0
        steps.append((low, high))
        remove(high)
        if low:
            remove(low)
    return tuple(steps)


def _zero_boundary(steps: tuple[tuple[int, int], ...], dim: int) -> bool:
    """Project oriented boundaries onto the critical dim-cells over Z."""
    projected: dict[int, dict[int, int]] = {}

    def boundary(high: int, skip: int = 0) -> dict[int, int]:
        result: Counter[int] = Counter()
        for i, bit in enumerate(_bit_list(high)):
            lower = high ^ bit
            if lower != skip:
                sign = (-1) ** i
                for face, count in projected.get(lower, {}).items():
                    result[face] += sign * count
        return {f: n for f, n in result.items() if n}

    # When a pair was removed, every other facet of its coface was
    # still present. Reverse order thus evaluates all dependencies first.
    for low, high in reversed(steps):
        if not low and high.bit_count() == dim + 1:
            projected[high] = {high: 1}
        elif low and low.bit_count() == dim + 1:
            bit = high ^ low
            signed_incidence = (-1) ** (high & (bit - 1)).bit_count()
            # In the Morse quotient, solve boundary(high)=0 for low.
            projected[low] = {
                f: -signed_incidence * n for f, n in boundary(high, low).items()
            }
    return all(
        not boundary(high)
        for low, high in steps
        if not low and high.bit_count() == dim + 2
    )


def _replay(
    faces: set[int], steps: tuple[tuple[int, int], ...]
) -> dict[int, int] | None:
    """Independently check actual live cofaces; ignore the search's heap/counts."""
    active = faces.copy()
    vertices = 0
    for f in faces:
        vertices |= f
    critical: Counter[int] = Counter()
    for low, high in steps:
        if high not in active or any(
            high | b in active for b in _bit_list(vertices ^ high)
        ):
            return None
        if low:
            if low not in active or low & high != low or (high ^ low).bit_count() != 1:
                return None
            if any(low | b in active for b in _bit_list(vertices ^ high)):
                return None
            active.remove(low)
        else:
            critical[high.bit_count() - 1] += 1
        active.remove(high)
    if active or critical.pop(0, 0) != 1:
        return None
    dims = sorted(critical)
    if len(dims) <= 1:
        return dict(critical)
    if (
        len(dims) == 2
        and dims[0] >= 2
        and dims[1] == dims[0] + 1
        and _zero_boundary(steps, dims[0])
    ):
        return dict(critical)
    return None


def _bounded_faces(facets: set[int], max_faces: int) -> set[int] | None:
    if not facets or max_faces < 1 or any(type(f) is not int or f <= 0 for f in facets):
        return None
    return _all_nonempty_faces(facets, max_faces)


def verify_integral_morse(
    facets: set[int], certificate: IntegralMorseCertificate, *, max_faces: int = 25000
) -> bool:
    """Replay the certificate against the supplied facet masks."""
    try:
        faces = _bounded_faces(facets, max_faces)
        if faces is None:
            return False
        spheres = _replay(faces, certificate.removals)
        return spheres is not None and spheres == certificate.spheres
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return False


def integral_morse_certificate(
    facets: set[int], *, max_faces: int = 25000
) -> IntegralMorseCertificate | None:
    """Find and replay an exact Morse certificate for a bounded complex."""
    faces = _bounded_faces(facets, max_faces)
    if faces is None:
        return None
    vertices = 0
    for facet in facets:
        vertices |= facet
    steps = _face_removals(faces, vertices)
    spheres = _replay(faces, steps)
    return None if spheres is None else IntegralMorseCertificate(steps, spheres)


def bipartite_shore_facets(graph: nx.Graph) -> set[int] | None:
    """A smaller complex whose suspension is the flag complex of graph.

    For a connected bipartite complement with sides A,B, cover the flag
    complex by the simplex A and the union of the stars of vertices of B.
    The latter is contractible: all nonempty intersections of these stars
    are cones (B spans a simplex and the complex is flag). The intersection
    with A has facets generated by N_graph(b) intersect A, b in B.
    This is Barmak's star-cluster proof of Jonsson's suspension theorem.
    Bit positions use the input graph's iteration order.
    """
    if (
        not graph
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
    ):
        return None
    complement = nx.complement(graph)
    if not nx.is_connected(complement) or not nx.is_bipartite(complement):
        return None
    left, right = nx.bipartite.sets(complement)
    if not left or not right:
        return None
    if len(left) > len(right):
        left, right = right, left
    index = {v: i for i, v in enumerate(graph)}
    generators = {
        sum(1 << index[v] for v in left.intersection(graph[b])) for b in right
    }
    return {
        f
        for f in generators
        if f and not any(f != g and f & g == f for g in generators)
    } or None
