"""Product recognition via independently checked strong-product splits.

Larrión, Pizaña and Villarroel-Flores (2013), Theorem 4.2, identifies
the simple-homotopy type of the strong product with the ordered product,
whose realization is the product of the realizations.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from pycliques.strong_products import (
    StrongProductSplit,
    find_strong_product_split,
    strong_product_factors,
)

from .homotopy_expressions import HomotopyType, ProductType, SphereType, WedgeType
from .homotopy_type import (
    _graph_exceeds_simplex_budget,
    homotopy_type_with_verdict,
)
from .presentation_types import (
    ComplexTypeCertificate,
    complex_type_certificate,
    verify_complex_type,
)
from .simplex import clique_complex


@dataclass(frozen=True)
class ProductTypeCertificate:
    """A verified graph split with certified homotopy types of its factors."""

    split: StrongProductSplit
    left: FactorCertificate
    right: FactorCertificate

    @property
    def homotopy_type(self) -> ProductType:
        factors: list[HomotopyType] = []
        for child in (self.left, self.right):
            term = _factor_type(child)
            if isinstance(term, ProductType):
                factors.extend(term.factors)
            elif term != WedgeType():
                factors.append(term)
        return ProductType(tuple(factors))

    @property
    def verdict(self) -> str:
        return r"\(" + self.homotopy_type.to_latex() + r"\)"


# A WedgeType leaf is a sphere-wedge verdict, replayed by the existing
# classifier; a ComplexTypeCertificate leaf is a cellular certificate.
FactorCertificate = WedgeType | ComplexTypeCertificate | ProductTypeCertificate


def _factor_type(certificate: FactorCertificate) -> HomotopyType:
    if isinstance(certificate, ProductTypeCertificate):
        return certificate.homotopy_type
    wedge = (
        certificate.homotopy_type
        if isinstance(certificate, ComplexTypeCertificate)
        else certificate
    )
    if len(wedge.summands) == 1 and wedge.summands[0][1] == 1:
        return wedge.summands[0][0]
    return wedge


def _sphere_wedge(graph: nx.Graph, max_simplices: int) -> WedgeType | None:
    verdict = homotopy_type_with_verdict(graph, max_simplices=max_simplices)
    wedge = verdict.proved_wedge
    if wedge is None:
        return None
    return WedgeType(
        tuple((SphereType(d), n) for d, n in sorted(wedge.spheres.items()) if n > 0)
    )


def _recognize_factor(
    graph: nx.Graph, max_simplices: int, max_partitions: int
) -> FactorCertificate | None:
    product = _recognize_product(graph, max_simplices, max_partitions)
    if product is not None:
        return product
    # Reuse the legacy classifier, including its existing Morse methods.
    wedge = _sphere_wedge(graph, max_simplices)
    if wedge is not None:
        return wedge
    if _graph_exceeds_simplex_budget(graph, max_simplices):
        return None
    return complex_type_certificate(clique_complex(graph))


def _recognize_product(
    graph: nx.Graph, max_simplices: int, max_partitions: int
) -> ProductTypeCertificate | None:
    split = find_strong_product_split(
        graph, max_vertices=len(graph), max_partitions=max_partitions
    )
    if split is None:
        return None
    left, right = strong_product_factors(graph, split)
    a = _recognize_factor(left, max_simplices, max_partitions)
    if a is None:
        return None
    b = _recognize_factor(right, max_simplices, max_partitions)
    if b is None:
        return None
    return ProductTypeCertificate(split, a, b)


def product_type_certificate(
    graph: nx.Graph,
    *,
    max_vertices: int = 256,
    max_simplices: int = 25_000,
    max_partitions: int = 128,
) -> ProductTypeCertificate | None:
    """Certify a product type of a connected graph's clique complex.

    Factorization is a bounded search on thin graphs; it may be inconclusive.
    Each split reduces graph order. The simplex budget applies to individual
    factor analyses, not the original graph's clique complex. In particular,
    the original product need never have its maximal cliques enumerated.
    """
    for bound in (max_vertices, max_simplices, max_partitions):
        if type(bound) is not int or bound < 1:
            raise ValueError("Product recognition budgets must be positive integers")
    if len(graph) > max_vertices:
        return None
    return _recognize_product(graph, max_simplices, max_partitions)


def _verify_factor(
    graph: nx.Graph, certificate: FactorCertificate, max_simplices: int
) -> bool:
    if isinstance(certificate, ProductTypeCertificate):
        return _verify_product(graph, certificate, max_simplices)
    if isinstance(certificate, WedgeType):
        return _sphere_wedge(graph, max_simplices) == certificate
    if isinstance(certificate, ComplexTypeCertificate):
        return not _graph_exceeds_simplex_budget(
            graph, max_simplices
        ) and verify_complex_type(clique_complex(graph), certificate)
    return False


def _verify_product(
    graph: nx.Graph, certificate: ProductTypeCertificate, max_simplices: int
) -> bool:
    left, right = strong_product_factors(graph, certificate.split)
    return _verify_factor(left, certificate.left, max_simplices) and _verify_factor(
        right, certificate.right, max_simplices
    )


def verify_product_type(
    graph: nx.Graph,
    certificate: ProductTypeCertificate,
    *,
    max_simplices: int = 25_000,
) -> bool:
    """Verify product coordinates and replay the factor homotopy proofs.

    Cellular leaves use the explicit certificate verifier. Legacy sphere
    leaves rerun the existing exact classifier and compare the proved wedge;
    no Betti-only or limited result is accepted. The product search is not
    rerun. A malformed or unrecognized certificate is rejected.
    """
    if type(max_simplices) is not int or max_simplices < 1:
        raise ValueError("max_simplices must be positive")
    try:
        return isinstance(certificate, ProductTypeCertificate) and _verify_product(
            graph, certificate, max_simplices
        )
    except (ValueError, TypeError, AttributeError, KeyError):
        return False
