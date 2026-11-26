"""Certified recognition of homotopy types of two-dimensional CW models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
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
from .indexed_relative_collapse import (
    indexed_relative_collapse,
    verify_indexed_relative_collapse,
)
from .relative_collapse import CollapseStatus
from .simplex import Simplex, SimplicialComplex, _stable_vertex_key


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

    The caller supplies a nonempty cyclically reduced word. Each label must
    occur exactly twice. Gluing its two sides identifies polygon corners in
    pairs; the resulting links are circles. The CW presentation identifies
    all polygon vertices, adding v-1 circle summands.
    """
    counts = Counter(map(abs, word))
    if set(counts) != generators or any(n != 2 for n in counts.values()):
        return None
    length = len(word)
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
    # A paired polygon is a closed connected surface, so the classification
    # theorem guarantees that the genus has the required parity and sign.
    if orientable:
        genus //= 2
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
            degree = gcd(*map(len, relators))
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
                if any(any(w.count(g) != w.count(-g) for g in block) for w in extra):
                    continue
            # A cyclically reduced nonempty paired word cannot give S^2:
            # its one-face cellulation would have a tree as its 1-skeleton,
            # with a leaf producing cyclically adjacent inverse letters.
            if surface == SurfaceType(1, False):
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
            isinstance(certificate.homotopy_type, WedgeType)
            and verify_presentation_reduction(presentation, certificate.reduction)
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


@dataclass(frozen=True)
class ComplexTypeCertificate:
    """A relative collapse followed by a certified presentation type."""

    core: SimplicialComplex
    collapse_pairs: tuple[tuple[Simplex, Simplex], ...]
    presentation: PresentationTypeCertificate

    @property
    def homotopy_type(self) -> WedgeType:
        return self.presentation.homotopy_type

    @property
    def verdict(self) -> str:
        return self.presentation.verdict


def complex_type_certificate(
    complex_: SimplicialComplex,
    *,
    max_target_facets: int = 25_000,
    max_bad_faces: int = 25_000,
    max_total_letters: int = 1_000_000,
) -> ComplexTypeCertificate | None:
    """Try a bounded collapse to dimension two, then recognize its CW type.

    For each maximal simplex choose an apex and retain all its apex
    triangles and their faces. The union includes the whole 1-skeleton.
    When distinct facets intersect in at most an edge, this target always
    admits a relative collapse. In general, the indexed relative-collapse
    verifier must establish the claimed collapse, or the search returns None.
    """
    if type(max_target_facets) is not int or max_target_facets < 1:
        raise ValueError("max_target_facets must be positive")
    if type(max_bad_faces) is not int or max_bad_faces < 0:
        raise ValueError("max_bad_faces must be nonnegative")
    pairs: tuple[tuple[Simplex, Simplex], ...] = ()
    core = complex_
    if complex_.dimension() > 2:
        facets: set[Simplex] = set()
        for face in complex_.facet_set:
            candidates: Iterable[Simplex]
            if len(face) <= 3:
                candidates = (face,)
            else:
                apex = min(face, key=_stable_vertex_key)
                candidates = (
                    Simplex((apex, u, v)) for u, v in combinations(face - {apex}, 2)
                )
            for candidate in candidates:
                facets.add(candidate)
                if len(facets) > max_target_facets:
                    return None
        core = SimplicialComplex(complex_.vertex_set, facet_set=facets)
        result = indexed_relative_collapse(complex_, core, max_bad_faces=max_bad_faces)
        if result.status is not CollapseStatus.PROVED:
            return None
        pairs = result.pairs
    if len(core.facet_set) > max_target_facets:
        return None
    presentation = two_complex_type_certificate(
        core, max_total_letters=max_total_letters
    )
    if presentation is None:
        return None
    return ComplexTypeCertificate(core, pairs, presentation)


def verify_complex_type(
    complex_: SimplicialComplex,
    certificate: ComplexTypeCertificate,
    *,
    max_bad_faces: int = 25_000,
) -> bool:
    """Verify both the relative collapse and the cellular recognition."""
    try:
        return verify_indexed_relative_collapse(
            complex_,
            certificate.core,
            certificate.collapse_pairs,
            max_bad_faces=max_bad_faces,
        ) and verify_two_complex_type(certificate.core, certificate.presentation)
    except (ValueError, TypeError, AttributeError):
        return False
