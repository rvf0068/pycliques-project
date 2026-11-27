"""Conservative comparison of finite simplicial homotopy types.

Positive answers require a verified simplicial homotopy-equivalence
certificate. Negative answers require a mismatch in an explicitly selected
homology invariant. Matching invariants without a certificate are
inconclusive.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .homology import (
    CoefficientField,
    HomologyBackend,
    HomologyResult,
    _normalize_field,
    simplicial_homology,
)
from .homology_backend import HomologyResourceLimit
from .simplex import SimplicialComplex
from .simplicial_maps import (
    SimplicialHomotopyEquivalenceCertificate,
    verify_simplicial_homotopy_equivalence,
)


class HomotopyComparisonStatus(Enum):
    """Strength of a comparison between two finite complexes."""

    PROVED_EQUIVALENT = "proved_equivalent"
    PROVED_NOT_EQUIVALENT = "proved_not_equivalent"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class HomotopyComparisonResult:
    """A conservative, auditable result of comparing two complexes.

    field_profiles stores (field, source_betti, target_betti) triples for
    every completed comparison. Profiles are reduced Betti numbers with
    trailing zeroes removed. A positive result stores the verified
    simplicial homotopy-equivalence certificate. A negative result is
    justified by a homology mismatch over at least one listed field.
    """

    status: HomotopyComparisonStatus
    reason: str
    field_profiles: tuple[
        tuple[CoefficientField, tuple[int, ...], tuple[int, ...]], ...
    ] = ()
    certificate: SimplicialHomotopyEquivalenceCertificate | None = None
    diagnostic: str | None = None

    @property
    def is_equivalent(self) -> bool | None:
        """Return True/False for proved outcomes and None if inconclusive."""
        if self.status is HomotopyComparisonStatus.PROVED_EQUIVALENT:
            return True
        if self.status is HomotopyComparisonStatus.PROVED_NOT_EQUIVALENT:
            return False
        return None


def _trim_profile(result: HomologyResult) -> tuple[int, ...]:
    profile = list(result.betti_numbers)
    while profile and profile[-1] == 0:
        profile.pop()
    return tuple(profile)


def compare_homotopy(
    source: SimplicialComplex,
    target: SimplicialComplex,
    *,
    certificate: SimplicialHomotopyEquivalenceCertificate | None = None,
    fields: Iterable[CoefficientField] = (2, "Q"),
    backend: HomologyBackend = "auto",
    max_faces: int = 100_000,
) -> HomotopyComparisonResult:
    """Compare finite simplicial complexes without overclaiming.

    A supplied certificate is independently verified first. If it verifies,
    the result is PROVED_EQUIVALENT and no homology is computed.

    Otherwise, exact reduced homology is compared over each requested field
    (by default GF(2) and Q). A mismatch over any one field proves that the
    complexes are not homotopy equivalent. If all requested profiles agree,
    the result is INCONCLUSIVE: homology agreement is not a homotopy
    equivalence certificate. Increase fields to test additional prime
    characteristics; no finite default field list detects all possible
    torsion.

    HomologyResourceLimit is converted into an inconclusive result. Other
    invalid inputs, including invalid coefficient fields, raise their usual
    TypeError or ValueError.
    """
    if not isinstance(source, SimplicialComplex) or not isinstance(
        target, SimplicialComplex
    ):
        raise TypeError("source and target must be SimplicialComplex instances")
    if not source.vertex_set or not target.vertex_set:
        raise ValueError("homotopy comparison requires nonempty realizations")

    selected_fields = tuple(_normalize_field(field) for field in fields)
    if not selected_fields:
        raise ValueError("fields must contain at least one coefficient field")
    if len(set(selected_fields)) != len(selected_fields):
        raise ValueError("fields must not contain duplicates")

    certificate_diagnostic = None
    if certificate is not None:
        if not isinstance(certificate, SimplicialHomotopyEquivalenceCertificate):
            raise TypeError("certificate must be a simplicial homotopy certificate")
        if verify_simplicial_homotopy_equivalence(source, target, certificate):
            return HomotopyComparisonResult(
                status=HomotopyComparisonStatus.PROVED_EQUIVALENT,
                reason=(
                    "A supplied simplicial homotopy-equivalence certificate "
                    "was independently verified."
                ),
                certificate=certificate,
            )
        certificate_diagnostic = (
            "The supplied homotopy-equivalence certificate did not verify."
        )

    profiles: list[tuple[CoefficientField, tuple[int, ...], tuple[int, ...]]] = []
    for field in selected_fields:
        try:
            source_result = simplicial_homology(
                source, field=field, reduced=True, backend=backend, max_faces=max_faces
            )
            target_result = simplicial_homology(
                target, field=field, reduced=True, backend=backend, max_faces=max_faces
            )
        except HomologyResourceLimit as error:
            return HomotopyComparisonResult(
                status=HomotopyComparisonStatus.INCONCLUSIVE,
                reason=(
                    "The homology comparison exceeded its configured resource limit."
                ),
                field_profiles=tuple(profiles),
                diagnostic=(
                    f"{certificate_diagnostic} {error}"
                    if certificate_diagnostic
                    else str(error)
                ),
            )

        source_profile = _trim_profile(source_result)
        target_profile = _trim_profile(target_result)
        profiles.append((field, source_profile, target_profile))
        if source_profile != target_profile:
            coefficient_label = "Q" if field == "Q" else f"GF({field})"
            return HomotopyComparisonResult(
                status=HomotopyComparisonStatus.PROVED_NOT_EQUIVALENT,
                reason=(
                    f"Reduced homology differs over {coefficient_label}; "
                    "homotopy-equivalent complexes have isomorphic homology "
                    "over every coefficient field."
                ),
                field_profiles=tuple(profiles),
                diagnostic=certificate_diagnostic,
            )

    return HomotopyComparisonResult(
        status=HomotopyComparisonStatus.INCONCLUSIVE,
        reason=(
            "The requested reduced homology profiles agree, but no "
            "homotopy-equivalence certificate was supplied and homology "
            "agreement alone does not prove homotopy equivalence."
        ),
        field_profiles=tuple(profiles),
        diagnostic=certificate_diagnostic,
    )
