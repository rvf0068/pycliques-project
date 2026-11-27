"""Unified, conservative homotopy-type classification entry point.

This module coordinates existing graph and simplicial-complex recognizers.
It does not replace their algorithms or infer homotopy type from homology.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import networkx as nx

from .homotopy_expressions import HomotopyType, SphereType, WedgeType
from .homotopy_results import HomotopyVerdict, WedgeOfSpheres
from .homotopy_type import homotopy_type_sc_with_verdict, homotopy_type_with_verdict
from .presentation_types import ComplexTypeCertificate, complex_type_certificate
from .product_types import (
    ProductTypeCertificate,
    product_type_certificate,
)
from .simplex import SimplicialComplex


class HomotopyClassificationStatus(Enum):
    """Strength of the information established by the classifier."""

    PROVED_TYPE = "proved_type"
    BETTI_ONLY = "betti_only"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class HomotopyClassification:
    """Result of a bounded homotopy-type classification attempt.

    Exactly one of the following is returned:
    - PROVED_TYPE: homotopy_type is a proved structured homotopy type.
    - BETTI_ONLY: betti_numbers are known, but no homotopy type was proved.
    - INCONCLUSIVE: configured limits prevented a conclusion, or the input
      lies outside the connected-realization scope currently supported by
      the structured HomotopyType model.

    A certificate is supplied when a dedicated certificate recognizer
    succeeded. A legacy HomotopyVerdict records the theorem strategy used by
    the established graph/complex cascade; it is not itself a replayable
    certificate.
    """

    status: HomotopyClassificationStatus
    reason: str
    homotopy_type: HomotopyType | None = None
    betti_numbers: tuple[int, ...] | None = None
    certificate: ComplexTypeCertificate | ProductTypeCertificate | None = None
    verdict: HomotopyVerdict | None = None

    @property
    def is_exact(self) -> bool:
        """Whether a homotopy type, rather than only homology, was proved."""
        return self.status is HomotopyClassificationStatus.PROVED_TYPE

    @property
    def has_replayable_certificate(self) -> bool:
        """Whether this result carries a dedicated, independently checkable certificate.

        Exact legacy cascade verdicts can prove a type without carrying a
        replayable certificate; inspect the verdict separately in that case.
        """
        return self.certificate is not None


def _wedge_type(wedge: WedgeOfSpheres) -> WedgeType:
    """Convert the legacy sphere-wedge representation to the structured API."""
    return WedgeType(
        tuple(
            (SphereType(dimension), multiplicity)
            for dimension, multiplicity in sorted(wedge.spheres.items())
            if multiplicity > 0
        )
    )


def _from_verdict(verdict: HomotopyVerdict) -> HomotopyClassification:
    if verdict.is_limited:
        return HomotopyClassification(
            HomotopyClassificationStatus.INCONCLUSIVE,
            verdict.reason,
            verdict=verdict,
        )
    if verdict.is_exact:
        return HomotopyClassification(
            HomotopyClassificationStatus.PROVED_TYPE,
            verdict.reason,
            homotopy_type=_wedge_type(verdict.wedge),
            verdict=verdict,
        )
    return HomotopyClassification(
        HomotopyClassificationStatus.BETTI_ONLY,
        verdict.reason,
        betti_numbers=tuple(verdict.wedge.to_betti()),
        verdict=verdict,
    )


def classify_homotopy(
    space: nx.Graph | SimplicialComplex,
    *,
    max_vertices: int | None = None,
    max_simplices: int | None = 25_000,
    max_total_letters: int = 1_000_000,
    recognize_products: bool = True,
    max_product_partitions: int = 128,
    dong_seed: int = 0,
    recursive_betti_states: int = 4096,
) -> HomotopyClassification:
    """Classify the homotopy type of a finite graph or simplicial complex.

    A NetworkX graph is interpreted through its clique complex. A
    SimplicialComplex is analyzed as given (its one-skeleton is never
    completed to a flag complex).

    The classifier first tries the dedicated cellular certificate for a
    simplicial complex, or bounded strong-product recognition for a graph
    when recognize_products is true. It then delegates to the existing
    graph/complex strategy cascade. Exact legacy wedge results are converted
    to the structured HomotopyType interface. Betti-only results are kept
    separate, and a resource-limit result is never presented as a type.

    The returned certificate can be independently checked with
    verify_complex_type or verify_product_type. A HomotopyVerdict is retained
    for the established strategy cascade, whose individual matching and
    reduction checks are performed by that algorithm.

    Parameters
    ----------
    space:
        A simple undirected loopless NetworkX graph or a nonempty
        SimplicialComplex.
    max_vertices, max_simplices:
        Resource limits forwarded to the established graph/complex cascade.
        Specialized certificate recognizers have separate fixed internal
        budgets and may establish a type without enumerating the full complex.
    max_total_letters:
        Bound for the cellular presentation recognizer.
    recognize_products:
        Whether to try the bounded strong-product factor recognizer first
        for graph inputs.
    max_product_partitions:
        Partition-search budget for product recognition.
    dong_seed, recursive_betti_states:
        Controls forwarded to the graph classifier.

    Notes
    -----
    Failure to recognize a type is not evidence that the space is not a
    wedge, contractible, or homotopy equivalent to another space.
    """
    if max_vertices is not None and (type(max_vertices) is not int or max_vertices < 0):
        raise ValueError("max_vertices must be a nonnegative integer or None")
    if max_simplices is not None and (
        type(max_simplices) is not int or max_simplices < 0
    ):
        raise ValueError("max_simplices must be a nonnegative integer or None")
    if type(max_total_letters) is not int or max_total_letters < 1:
        raise ValueError("max_total_letters must be a positive integer")
    if type(max_product_partitions) is not int or max_product_partitions < 1:
        raise ValueError("max_product_partitions must be a positive integer")
    if type(recognize_products) is not bool:
        raise TypeError("recognize_products must be boolean")

    if isinstance(space, SimplicialComplex):
        if not space.vertex_set:
            raise ValueError("Homotopy classification requires a nonempty realization")
        if max_vertices is not None and len(space.vertex_set) > max_vertices:
            return HomotopyClassification(
                HomotopyClassificationStatus.INCONCLUSIVE,
                f"vertex limit {max_vertices} exceeded",
            )
        if not nx.is_connected(space.one_skeleton_graph()):
            return HomotopyClassification(
                HomotopyClassificationStatus.INCONCLUSIVE,
                "The realization is disconnected; the current HomotopyType "
                "expressions represent connected based homotopy types.",
            )
        certificate = complex_type_certificate(
            space,
            max_target_facets=25_000,
            max_bad_faces=25_000,
            max_total_letters=max_total_letters,
        )
        if certificate is not None:
            return HomotopyClassification(
                HomotopyClassificationStatus.PROVED_TYPE,
                "A relative-collapse and cellular-presentation certificate "
                "recognized the homotopy type.",
                homotopy_type=certificate.homotopy_type,
                certificate=certificate,
            )
        return _from_verdict(
            homotopy_type_sc_with_verdict(
                space,
                max_vertices=max_vertices,
                max_simplices=max_simplices,
                dong_seed=dong_seed,
            )
        )

    if not isinstance(space, nx.Graph):
        raise TypeError("space must be a NetworkX graph or SimplicialComplex")
    if space.is_directed() or space.is_multigraph() or nx.number_of_selfloops(space):
        raise ValueError("graph input must be simple, undirected, and loopless")
    if not space:
        raise ValueError("Homotopy classification requires a nonempty realization")
    if max_vertices is not None and space.number_of_nodes() > max_vertices:
        return HomotopyClassification(
            HomotopyClassificationStatus.INCONCLUSIVE,
            f"vertex limit {max_vertices} exceeded",
        )
    if not nx.is_connected(space):
        return HomotopyClassification(
            HomotopyClassificationStatus.INCONCLUSIVE,
            "The realization is disconnected; the current HomotopyType "
            "expressions represent connected based homotopy types.",
        )

    if recognize_products:
        product_certificate = product_type_certificate(
            space,
            max_vertices=256 if max_vertices is None else max_vertices,
            max_simplices=25_000,
            max_partitions=max_product_partitions,
        )
        if product_certificate is not None:
            return HomotopyClassification(
                HomotopyClassificationStatus.PROVED_TYPE,
                "A strong-product split and certified factor types "
                "recognized the homotopy type.",
                homotopy_type=product_certificate.homotopy_type,
                certificate=product_certificate,
            )

    return _from_verdict(
        homotopy_type_with_verdict(
            space,
            max_vertices=max_vertices,
            max_simplices=max_simplices,
            dong_seed=dong_seed,
            recursive_betti_states=recursive_betti_states,
        )
    )
