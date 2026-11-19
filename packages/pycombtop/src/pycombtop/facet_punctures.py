"""Bounded, independently replayable facet-puncture wedge certificates.

If X collapses to R, and R with m maximal d-facets removed collapses
to a vertex, X is homotopy equivalent to the wedge of m d-spheres.
A failed search is inconclusive, not evidence of nonexistence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
from itertools import combinations

from .simplex import Simplex, SimplicialComplex


@dataclass(frozen=True)
class FacetPunctureWedgeCertificate:
    """Each collapse pair is (free codimension-one face, maximal coface)."""

    dimension: int
    removed_facets: tuple[Simplex, ...]
    initial_pairs: tuple[tuple[Simplex, Simplex], ...]
    final_pairs: tuple[tuple[Simplex, Simplex], ...]

    @property
    def spheres(self) -> dict[int, int]:
        """The exact homotopy type, provided verification succeeds."""
        return {self.dimension: len(self.removed_facets)}


def _stable_key(vertex):
    return (type(vertex).__module__, type(vertex).__qualname__, repr(vertex))


def _encode_complex(complex_: SimplicialComplex, max_faces: int):
    if max_faces < 1:
        raise ValueError("max_faces must be positive")
    facets = tuple(complex_.facet_set)
    if not facets or any(not f for f in facets):
        return None
    vertices = sorted(set().union(*facets), key=_stable_key)
    if len(vertices) > 64:
        return None
    positions = {v: 1 << i for i, v in enumerate(vertices)}
    faces: set[int] = set()
    for facet in facets:
        if (1 << len(facet)) - 1 > max_faces:
            return None
        mask = sum(positions[v] for v in facet)
        sub = mask
        while sub:
            faces.add(sub)
            if len(faces) > max_faces:
                return None
            sub = (sub - 1) & mask
    return vertices, faces


def _bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit
        mask -= bit


def _boundary(face: int):
    for bit in _bits(face):
        sub = face - bit
        if sub:
            yield sub


def _greedy_collapse(
    original: set[int], n_vertices: int
) -> tuple[set[int], list[tuple[int, int]]]:
    """Incremental free-face search; certificates are checked independently."""
    faces = set(original)
    upper: Counter[int] = Counter()
    for high in faces:
        for low in _boundary(high):
            upper[low] += 1
    queue: list[tuple[int, int, int]] = []

    def enqueue(high: int):
        if high not in faces or upper[high] or high.bit_count() < 2:
            return
        for low in _boundary(high):
            if low in faces and upper[low] == 1:
                heapq.heappush(queue, (-high.bit_count(), high, low))
                return

    for face in faces:
        enqueue(face)

    pairs: list[tuple[int, int]] = []
    while queue:
        _, high, low = heapq.heappop(queue)
        if (
            high not in faces
            or low not in faces
            or upper[high] != 0
            or upper[low] != 1
            or low.bit_count() + 1 != high.bit_count()
            or low & high != low
        ):
            continue
        faces.difference_update((high, low))
        pairs.append((low, high))
        for removed in (high, low):
            for sub in _boundary(removed):
                if sub not in faces:
                    continue
                upper[sub] -= 1
                if upper[sub] < 0:
                    raise AssertionError("Negative coface count")
                if upper[sub] == 0:
                    enqueue(sub)
                elif upper[sub] == 1:
                    for i in range(n_vertices):
                        bit = 1 << i
                        if not sub & bit and (sub | bit) in faces:
                            enqueue(sub | bit)
                            break
    return faces, pairs


def _replay(
    original: set[int],
    pairs: tuple[tuple[Simplex, Simplex], ...],
    vertices: list,
) -> set[int] | None:
    """Validate every free pair directly in the complete face poset."""
    faces = set(original)
    positions = {v: 1 << i for i, v in enumerate(vertices)}
    if not isinstance(pairs, (tuple, list)):
        return None
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return None
        try:
            low, high = (sum(positions[v] for v in f) for f in pair)
        except (TypeError, ValueError, KeyError):
            return None
        if (
            low not in faces
            or high not in faces
            or low.bit_count() < 1
            or low.bit_count() + 1 != high.bit_count()
            or high & low != low
        ):
            return None
        if any(
            (high | (1 << i)) in faces
            for i in range(len(vertices))
            if not high & (1 << i)
        ):
            return None
        if sum(
            (low | (1 << i)) in faces
            for i in range(len(vertices))
            if not low & (1 << i)
        ) != 1:
            return None
        faces.difference_update((low, high))
    return faces


def _top_betti_mod2(faces: set[int], top_size: int) -> int:
    """Exact top Betti number: there is no boundary from above."""
    top = sorted(f for f in faces if f.bit_count() == top_size)
    lower = sorted(f for f in faces if f.bit_count() == top_size - 1)
    row = {f: i for i, f in enumerate(lower)}
    pivots: dict[int, int] = {}
    for face in top:
        column = sum(1 << row[f] for f in _boundary(face))
        while column:
            i = column.bit_length() - 1
            if i not in pivots:
                pivots[i] = column
                break
            column ^= pivots[i]
    return len(top) - len(pivots)


def _decode(mask: int, vertices: list) -> Simplex:
    return Simplex(vertices[i] for i in range(len(vertices)) if mask & (1 << i))


def verify_facet_puncture_wedge(
    complex_: SimplicialComplex,
    certificate: FacetPunctureWedgeCertificate,
) -> bool:
    """Independently certify a wedge of spheres; reject invalid witnesses."""
    if not isinstance(certificate, FacetPunctureWedgeCertificate):
        return False
    try:
        encoded = _encode_complex(complex_, max_faces=500_000)
        if encoded is None or certificate.dimension < 1:
            return False
        vertices, original = encoded
        core = _replay(original, certificate.initial_pairs, vertices)
        if not core or not certificate.removed_facets:
            return False
        positions = {v: 1 << i for i, v in enumerate(vertices)}
        removed = []
        for facet in certificate.removed_facets:
            mask = sum(positions[v] for v in facet)
            if (
                mask not in core
                or mask.bit_count() != certificate.dimension + 1
                or any(
                    (mask | (1 << i)) in core
                    for i in range(len(vertices))
                    if not mask & (1 << i)
                )
            ):
                return False
            removed.append(mask)
        if len(removed) != len(set(removed)):
            return False
        # Only maximal facets are deleted, so boundaries remain.
        terminal = _replay(
            core - set(removed), certificate.final_pairs, vertices
        )
        return (
            terminal is not None
            and len(terminal) == 1
            and next(iter(terminal)).bit_count() == 1
        )
    except (TypeError, ValueError, KeyError, OverflowError):
        return False


def facet_puncture_wedge_certificate(
    complex_: SimplicialComplex,
    *,
    max_faces: int = 10_000,
    max_candidates: int = 5_000,
    max_removed_facets: int = 6,
) -> FacetPunctureWedgeCertificate | None:
    """Find a replayable wedge of equal-dimensional spheres, if possible.

    A successful puncture of m top d-facets requires
    m = beta_d(R; F_2), so a rank calculation reduces the subset
    enumeration to one cardinality. Search budgets affect completeness
    only. Up to 64 vertices and max_faces nonempty faces are supported.
    """
    if min(max_faces, max_candidates, max_removed_facets) < 1:
        raise ValueError("Budgets must be positive")
    encoded = _encode_complex(complex_, max_faces)
    if encoded is None:
        return None
    vertices, original = encoded
    core, initial_pairs = _greedy_collapse(original, len(vertices))
    top_size = max(face.bit_count() for face in core)
    if top_size < 2:
        return None
    required = _top_betti_mod2(core, top_size)
    if required < 1 or required > max_removed_facets:
        return None
    top = sorted(f for f in core if f.bit_count() == top_size)
    for i, chosen in enumerate(combinations(top, required)):
        if i >= max_candidates:
            break
        terminal, final_pairs = _greedy_collapse(
            core - set(chosen), len(vertices)
        )
        if len(terminal) != 1 or next(iter(terminal)).bit_count() != 1:
            continue
        witness = FacetPunctureWedgeCertificate(
            dimension=top_size - 1,
            removed_facets=tuple(_decode(f, vertices) for f in chosen),
            initial_pairs=tuple(
                (_decode(low, vertices), _decode(high, vertices))
                for low, high in initial_pairs
            ),
            final_pairs=tuple(
                (_decode(low, vertices), _decode(high, vertices))
                for low, high in final_pairs
            ),
        )
        if verify_facet_puncture_wedge(complex_, witness):
            return witness
    return None
