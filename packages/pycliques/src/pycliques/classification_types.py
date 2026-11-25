"""Clique behavior, reference status and classification witnesses."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum, auto

import networkx as nx

from .clockwork import ClockworkStructure
from .clockwork_search import InducedClockworkWitness
from .retractions import SpecialOctahedronWitness


class Verdict(Enum):
    """Classification outcome for a graph's clique behavior."""

    CONVERGENT = auto()
    DIVERGENT = auto()
    INDETERMINATE = auto()


class ReferenceStatus(Enum):
    """Whether a registered reference graph is proved or only conjectured."""

    PROVEN = auto()
    CONJECTURED = auto()


@dataclass(frozen=True)
class Certificate:
    """Structured provenance for a classification decision.

    An inference rule returns ``None`` when it establishes no verdict.  An
    ``INDETERMINATE`` result is reserved for a conditional conclusion or for
    the global classifier after all applicable rules have been exhausted.
    """

    rule: str
    target: nx.Graph | None = None
    target_status: str | None = None
    target_label: str | None = None
    map: tuple[dict, dict] | dict | None = None
    edge: tuple[Hashable, Hashable] | None = None
    m: int | None = None
    n: int | None = None
    radius: int | None = None
    source: nx.Graph | None = None
    source_coaffination: dict[Hashable, Hashable] | None = None
    target_coaffination: dict[Hashable, Hashable] | None = None
    theorem_base_graph: nx.Graph | None = None
    suspension_vertices: tuple[Hashable, Hashable] | None = None
    base_graph: nx.Graph | None = None
    base_coaffination: dict[Hashable, Hashable] | None = None
    clockwork_structure: ClockworkStructure | None = None
    iterate_index: int | None = None
    special_octahedron: SpecialOctahedronWitness | None = None

    @classmethod
    def from_special_octahedron(
        cls, source: nx.Graph, index: int, witness: SpecialOctahedronWitness
    ) -> Certificate:
        """Record the witness on the indexed pared iterate."""
        return cls(
            "special_octahedron",
            source=source,
            target=source,
            target_status="proven_divergent",
            iterate_index=index,
            special_octahedron=witness,
        )

    @classmethod
    def from_retraction(
        cls,
        source: nx.Graph,
        target: nx.Graph,
        index: int,
        maps: tuple[dict, dict],
        *,
        rule: str = "retracts",
        label: str | None = None,
    ) -> Certificate:
        """Record a projection and section onto a proved divergent target."""
        return cls(
            rule,
            source=source,
            target=target,
            target_status="proven_divergent",
            iterate_index=index,
            map=maps,
            target_label=label,
        )


@dataclass(frozen=True)
class CliqueBehavior:
    """Result of classifying a graph under clique-graph iteration.

    ``INDETERMINATE`` means that the available sufficient tests did not
    decide the behavior within the configured limits.  It is not a
    mathematical assertion that the graph is neither convergent nor
    divergent.

    Individual inference rules return ``None`` when they establish no
    verdict.  ``INDETERMINATE`` is returned by the global classifier only
    after the applicable rules have been exhausted, or when a conclusion is
    explicitly conditional on a conjecture.

    .. rubric:: Attributes

    verdict : Verdict
        The classification supplied by the available tests.
    reason : str
        Explanation of the deciding test or why classification was
        indeterminate.
    iterations_checked : int
        Number of iterated clique graphs inspected, including the input.
    bound_exceeded : bool
        Whether a clique-graph computation exceeded the clique bound.
    pared_graph : networkx.Graph
        Completely pared copy of the input graph used by the classifier.
    certificate : Certificate | None
        Structured justification for the inference rule that established the
        verdict, when available.
    observations : tuple[InducedClockworkWitness, ...]
        Verified induced divergent clockworks found by opt-in searches. These
        structural observations do not, on their own, establish divergence.
    """

    verdict: Verdict
    reason: str
    iterations_checked: int
    bound_exceeded: bool
    pared_graph: nx.Graph
    certificate: Certificate | None = None
    observations: tuple[InducedClockworkWitness, ...] = ()
