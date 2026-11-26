"""Certified recognition of homotopy types of two-dimensional CW models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import gcd

import networkx as nx

from .fundamental_group import (
    CellularPresentation,
    PresentationReduction,
    _cellular_free_reduce,
    cellular_presentation,
    reduce_cellular_presentation,
    verify_presentation_reduction,
)
from .homotopy_expressions import (
    HomotopyType,
    MooreSpaceType,
    SphereType,
    SurfaceType,
    WedgeType,
)
from .simplex import SimplicialComplex


@dataclass(frozen=True)
class PresentationTypeCertificate:
    """A replayable presentation reduction and its recognized homotopy type."""

    reduction: PresentationReduction
    homotopy_type: WedgeType

    @property
    def verdict(self) -> str:
        return r"\(" + self.homotopy_type.to_latex() + r"\)"


def _cyclic_reduce(word: tuple[int, ...]) -> tuple[int, ...]:
    reduced = _cellular_free_reduce(word)
    i, j = 0, len(reduced)
    while j - i > 1 and reduced[i] == -reduced[j - 1]:
        i += 1
        j -= 1
    return reduced[i:j]


def _polygon_surface(
    word: tuple[int, ...], generators: set[int]
) -> tuple[SurfaceType, int] | None:
    """Classify the surface of a paired polygon and count its vertex classes.

    Each label must occur exactly twice. Gluing its two sides identifies
    polygon corners in pairs; the resulting links are circles. The CW
    presentation identifies all polygon vertices, adding v-1 circle summands.
    """
    counts = Counter(map(abs, word))
    if set(counts) != generators or any(n != 2 for n in counts.values()):
        return None
    length = len(word)
    if not length:
        return None
    corners = nx.utils.UnionFind(range(length))
    first: dict[int, int] = {}
    orientable = True
    for i, letter in enumerate(word):
        g = abs(letter)
        if g not in first:
            first[g] = i
            continue
        j = first[g]
        if letter == word[j]:
            orientable = False
            corners.union(i, j)
            corners.union((i + 1) % length, (j + 1) % length)
        else:
            corners.union(i, (j + 1) % length)
            corners.union((i + 1) % length, j)
    vertices = len({corners[i] for i in range(length)})
    chi = vertices - len(generators) + 1
    genus = 2 - chi
    if orientable:
        if genus < 0 or genus % 2:
            return None
        genus //= 2
    elif genus < 1:
        return None
    return SurfaceType(genus, orientable), vertices


def _classify_reduced(presentation: CellularPresentation) -> WedgeType | None:
    """Recognize disjoint generator blocks; never infer type from H1 alone."""
    words = [_cyclic_reduce(word) for word in presentation.relators]
    counts: Counter[HomotopyType] = Counter()

    def add(term: HomotopyType, count: int = 1) -> None:
        if count:
            counts[term] += count

    add(SphereType(2), sum(not word for word in words))
    supports = [(word, set(map(abs, word))) for word in words if word]
    graph = nx.Graph()
    graph.add_nodes_from(presentation.generators)
    for _, support in supports:
        root = min(support)
        graph.add_edges_from((root, g) for g in support if g != root)

    for block in nx.connected_components(graph):
        relators = [word for word, support in supports if support <= block]
        if len(block) == 1:
            # Cell slides realize the Euclidean algorithm on attaching degrees.
            degree = gcd(*(abs(sum(1 if x > 0 else -1 for x in w)) for w in relators))
            if not degree:
                add(SphereType(1))
                add(SphereType(2), len(relators))
            else:
                if degree > 1:
                    add(MooreSpaceType(degree))
                add(SphereType(2), len(relators) - 1)
            continue

        for i, word in enumerate(relators):
            polygon = _polygon_surface(word, block)
            if polygon is None:
                continue
            surface, vertices = polygon
            extra = relators[:i] + relators[i + 1 :]
            if extra:
                # Only use exponent sums for the actual one-vertex torus.
                if surface != SurfaceType(1) or vertices != 1:
                    continue
                if any(
                    any(w.count(g) != w.count(-g) for g in block) for w in extra
                ):
                    continue
            if surface == SurfaceType(0):
                add(SphereType(2))
            elif surface == SurfaceType(1, False):
                add(MooreSpaceType(2))
            else:
                add(surface)
            add(SphereType(1), vertices - 1)
            add(SphereType(2), len(extra))
            break
        else:
            return None

    return WedgeType(tuple(sorted(counts.items(), key=lambda item: item[0].to_latex())))


def presentation_type_certificate(
    presentation: CellularPresentation, *, max_total_letters: int = 1_000_000
) -> PresentationTypeCertificate | None:
    """Recognize a wedge of circles, spheres, Moore spaces and closed surfaces.

    A failed or resource-limited search is inconclusive.
    """
    reduction = reduce_cellular_presentation(
        presentation, max_total_letters=max_total_letters
    )
    if reduction.limited:
        return None
    homotopy_type = _classify_reduced(reduction.reduced)
    if homotopy_type is None:
        return None
    return PresentationTypeCertificate(reduction, homotopy_type)


def verify_presentation_type(
    presentation: CellularPresentation, certificate: PresentationTypeCertificate
) -> bool:
    """Replay the elimination and independently check the recognized type."""
    try:
        return (
            verify_presentation_reduction(presentation, certificate.reduction)
            and _classify_reduced(certificate.reduction.reduced)
            == certificate.homotopy_type
        )
    except (ValueError, TypeError, AttributeError):
        return False


def two_complex_type_certificate(
    complex_: SimplicialComplex, *, max_total_letters: int = 1_000_000
) -> PresentationTypeCertificate | None:
    """Recognize the type of a connected nonempty complex of dimension <=2.

    Higher-dimensional input is rejected by cellular_presentation. A caller
    may first apply a homotopy-preserving collapse of the whole complex.
    """
    return presentation_type_certificate(
        cellular_presentation(complex_), max_total_letters=max_total_letters
    )


def verify_two_complex_type(
    complex_: SimplicialComplex, certificate: PresentationTypeCertificate
) -> bool:
    """Reconstruct the cellular presentation and verify its certificate."""
    try:
        return verify_presentation_type(cellular_presentation(complex_), certificate)
    except (ValueError, TypeError, AttributeError):
        return False
