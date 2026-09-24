"""Determine the clique behavior of small graphs."""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from itertools import combinations
from pathlib import Path
from typing import cast

import networkx as nx
from rich.logging import RichHandler

from pycliques import __version__
from pycliques.cliques import clique_graph
from pycliques.clockwork import is_clique_divergent_clockwork, recognize_clockwork
from pycliques.clockwork_pairs import (
    candidate_target_coaffinations,
    clockwork_coaffine_pair,
    find_pair_morphism,
)
from pycliques.coaffinations import CoaffinePair, coaffinations
from pycliques.cutpoints import (
    contract_local_bridge,
    edge_in_triangle,
    edges_in_no_triangle,
    inverse_cutpoint_extensions,
    is_admissible_inverse_extension,
    local_bridges,
    remove_edge_not_in_triangle,
)
from pycliques.dominated import completely_pared_graph, find_dominated_vertex
from pycliques.helly import is_clique_helly
from pycliques.named import complement_of_cycle, snub_disphenoid, suspension_of_cycle
from pycliques.retractions import retracts, special_octahedra_dimension

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict enum and CliqueSequence
# ---------------------------------------------------------------------------


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
    suspension_vertices: tuple[Hashable, Hashable] | None = None
    base_graph: nx.Graph | None = None
    base_coaffination: dict[Hashable, Hashable] | None = None


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
    """

    verdict: Verdict
    reason: str
    iterations_checked: int
    bound_exceeded: bool
    pared_graph: nx.Graph
    certificate: Certificate | None = None


#: Type alias for one inference rule's result.
ClassifierResult = tuple[Verdict, str, Certificate | None]
#: Type alias for an inference rule. ``None`` means this rule proved nothing.
Classifier = Callable[["CliqueSequence"], ClassifierResult | None]


def suspension_bases(
    graph: nx.Graph,
) -> Iterator[tuple[Hashable, Hashable, nx.Graph]]:
    """Yield connected bases of genuine suspension decompositions.

    A yielded pair ``(u, v, H)`` satisfies
    ``graph = H * complement(K_2)`` structurally: ``u`` and ``v`` are
    nonadjacent, universal outside the pair, and deleting them leaves the
    connected graph ``H``.  The vertices may have arbitrary hashable labels.
    """
    vertices = set(graph)
    for u, v in combinations(vertices, 2):
        if graph.has_edge(u, v):
            continue
        base_vertices = vertices - {u, v}
        if set(graph.neighbors(u)) != base_vertices:
            continue
        if set(graph.neighbors(v)) != base_vertices:
            continue
        base = graph.subgraph(base_vertices).copy()
        if base.number_of_nodes() > 0 and nx.is_connected(base):
            yield u, v, base


def _classify_suspension_2_coaffination(
    graph: nx.Graph,
) -> ClassifierResult | None:
    """Apply Theorem 4.6 to a suspension with a 2-coaffine base."""
    for u, v, base in suspension_bases(graph):
        for tau in coaffinations(base, 2):
            return (
                Verdict.DIVERGENT,
                (
                    "Theorem 4.6: graph is the suspension of a connected graph "
                    "H admitting a 2-coaffination; hence it is expansive and "
                    "therefore clique divergent"
                ),
                Certificate(
                    rule="theorem_4_6_suspension_2_coaffination",
                    target=graph,
                    target_status="proven_divergent",
                    suspension_vertices=(u, v),
                    base_graph=base,
                    base_coaffination=tau,
                    radius=2,
                ),
            )
    return None


class CliqueSequence:
    """Lazily computed, cached sequence of iterated pared clique graphs.

    ``seq[0]`` is the original (pared) graph. ``seq[i]`` for *i > 0* is
    the completely-pared clique graph of ``seq[i-1]``. Each level is
    computed at most once.

    .. rubric:: Parameters

    graph : networkx.Graph
        Starting (already pared) graph.
    bound : int
        Maximum number of cliques before aborting (default 30).

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.small import CliqueSequence
    >>> seq = CliqueSequence(nx.octahedral_graph())
    >>> seq[0].order()
    6

    """

    def __init__(
        self,
        graph: nx.Graph,
        bound: int = 30,
        on_iterate: Callable[[int], None] | None = None,
    ) -> None:
        self._graphs: list[nx.Graph] = [graph]
        self._bound = bound
        self._on_iterate = on_iterate
        self._exhausted = False

    @property
    def graph_count(self) -> int:
        """Return the number of iterates currently cached."""
        return len(self._graphs)

    @property
    def exhausted(self) -> bool:
        """Return whether generating another iterate has failed."""
        return self._exhausted

    def __getitem__(self, i: int) -> nx.Graph | None:
        """Return the *i*-th iterated pared clique graph, or ``None``."""
        while len(self._graphs) <= i and not self._exhausted:
            if self._on_iterate is not None:
                self._on_iterate(len(self._graphs))
            kg = clique_graph(self._graphs[-1], self._bound)
            if kg is None:
                self._exhausted = True
                return None
            self._graphs.append(completely_pared_graph(kg))
        if i < len(self._graphs):
            return self._graphs[i]
        return None


# ---------------------------------------------------------------------------
# Classifier functions
# ---------------------------------------------------------------------------

_MAX_ITERATIONS = 9
_REFERENCE_GRAPHS: dict[bytes, tuple[nx.Graph, ReferenceStatus, str | None]] = {}


def _canonical_reference_key(graph: nx.Graph) -> bytes:
    """Return a canonical graph6 key for a graph while preserving arbitrary labels."""
    numbered = nx.convert_node_labels_to_integers(graph)
    return cast(bytes, nx.to_graph6_bytes(numbered, header=False))


def register_reference_graph(
    graph: nx.Graph,
    *,
    status: ReferenceStatus = ReferenceStatus.PROVEN,
    label: str | None = None,
) -> None:
    """Register a graph whose clique behavior is known independently of the classifier.

    The status distinguishes a proven divergent reference graph from a merely
    conjectured one. The classifier uses the same inference rules for every
    registered graph, so future reference facts can be added without branching
    on individual graph names.

    label : str, optional
        Human-readable identifier for the reference graph, used in certificates
        and saved indeterminate-file metadata.
    """
    _REFERENCE_GRAPHS[_canonical_reference_key(graph)] = (graph.copy(), status, label)


def clear_reference_graphs() -> None:
    """Clear all registered divergent reference graphs."""
    _REFERENCE_GRAPHS.clear()


register_reference_graph(
    snub_disphenoid(),
    status=ReferenceStatus.CONJECTURED,
    label="snub_disphenoid",
)

register_reference_graph(
    nx.octahedral_graph(),
    status=ReferenceStatus.PROVEN,
    label="octahedral_graph",
)


def _reference_match(
    graph: nx.Graph,
) -> tuple[nx.Graph, ReferenceStatus, str | None] | None:
    """Return the first registered reference graph isomorphic to *graph*, if any."""
    for ref_graph, status, label in _REFERENCE_GRAPHS.values():
        if nx.is_isomorphic(graph, ref_graph):
            return ref_graph, status, label
    return None


def _summarize_reference_status(status: ReferenceStatus) -> str:
    """Render the reference status in the certificate vocabulary used by the API."""
    if status is ReferenceStatus.PROVEN:
        return "proven_divergent"
    return "conjectured_divergent"


def _classify_reference_graph(
    graph: nx.Graph,
) -> ClassifierResult | None:
    """Classify a graph directly from a registered reference fact, if present."""
    match = _reference_match(graph)
    if match is None:
        return None
    ref_graph, status, label = match
    if status is ReferenceStatus.PROVEN:
        return (
            Verdict.DIVERGENT,
            "registered reference graph is proven clique divergent",
            Certificate(
                rule="reference",
                target=ref_graph,
                target_status="proven_divergent",
                target_label=label,
            ),
        )
    return (
        Verdict.INDETERMINATE,
        (
            "registered reference graph is conjectured clique divergent; "
            "conditional on it being clique divergent"
        ),
        Certificate(
            rule="reference",
            target=ref_graph,
            target_status="conjectured_divergent",
            target_label=label,
        ),
    )


def _classify_reference_dependency(
    graph: nx.Graph,
) -> ClassifierResult | None:
    """Classify by a reference-dependent retraction, or return ``None``.

    ``None`` means this inference rule did not establish a verdict; it does
    not make the overall graph indeterminate.
    """
    for ref_graph, status, label in _REFERENCE_GRAPHS.values():
        retraction = retracts(graph, ref_graph)
        if isinstance(retraction, tuple):
            if status is ReferenceStatus.PROVEN:
                return (
                    Verdict.DIVERGENT,
                    "retracts to a proven divergent reference graph",
                    Certificate(
                        rule="retracts",
                        target=ref_graph,
                        target_status="proven_divergent",
                        target_label=label,
                        map=retraction,
                    ),
                )
            return (
                Verdict.INDETERMINATE,
                (
                    "retracts to a conjectured divergent reference graph; "
                    "conditional on the target being clique divergent"
                ),
                Certificate(
                    rule="retracts",
                    target=ref_graph,
                    target_status="conjectured_divergent",
                    target_label=label,
                    map=retraction,
                ),
            )
    return None


def _classify_local_bridge(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
) -> ClassifierResult | None:
    """Classify by Theorem 6.1, or return ``None`` when it proves nothing.

    A successful certificate records the contracted edge.  Failure to classify
    a contracted target falls through to the next inference rule.
    """
    for u, v in local_bridges(graph):
        h = contract_local_bridge(graph, u, v)
        h_behavior = classify_clique_behavior(
            h, tries=tries, bound=bound, extension_budget=extension_budget
        )
        if h_behavior.verdict is Verdict.CONVERGENT:
            return (
                Verdict.CONVERGENT,
                "contracting a local bridge yields a convergent graph by Theorem 6.1",
                Certificate(
                    rule="local_bridge",
                    target=h,
                    target_status="proven_convergent",
                    target_label=h_behavior.certificate.target_label
                    if h_behavior.certificate is not None
                    else None,
                    edge=(u, v),
                ),
            )
        if h_behavior.verdict is Verdict.DIVERGENT:
            return (
                Verdict.DIVERGENT,
                "contracting a local bridge yields a divergent graph by Theorem 6.1",
                Certificate(
                    rule="local_bridge",
                    target=h,
                    target_status="proven_divergent",
                    target_label=h_behavior.certificate.target_label
                    if h_behavior.certificate is not None
                    else None,
                    edge=(u, v),
                ),
            )
        if (
            h_behavior.certificate is not None
            and h_behavior.certificate.target_status == "conjectured_divergent"
        ):
            return (
                Verdict.INDETERMINATE,
                (
                    "contracting a local bridge yields a conjectured divergent "
                    "graph; conditional on the target being clique divergent "
                    "by Theorem 6.1"
                ),
                Certificate(
                    rule="local_bridge",
                    target=h_behavior.certificate.target or h,
                    target_status="conjectured_divergent",
                    target_label=h_behavior.certificate.target_label,
                    edge=(u, v),
                ),
            )
    return None


def classify_local_bridge(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
) -> ClassifierResult | None:
    """Classify clique behavior by contracting local bridges according to Theorem 6.1.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries : int, optional
        Maximum number of iterates to inspect (default: 9).
    bound : int, optional
        Maximum number of cliques allowed at each iteration (default: 30).

    .. rubric:: Returns

    tuple[Verdict, str, Certificate | None] | None
        A tuple of ``(verdict, reason, certificate)`` if a local bridge exists
        and the contracted graph can be classified; ``None`` if this rule
        establishes no verdict.
    """
    return _classify_local_bridge(graph, tries=tries, bound=bound)


def _classify_non_triangle_edge(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
) -> ClassifierResult | None:
    """Classify by Theorem 6.2, or return ``None`` when it proves nothing.

    A successful certificate records the deleted edge.  Failure to classify a
    deleted-edge target falls through to the next inference rule.
    """
    for u, v in edges_in_no_triangle(graph):
        h = remove_edge_not_in_triangle(graph, u, v)
        h_behavior = classify_clique_behavior(
            h, tries=tries, bound=bound, extension_budget=extension_budget
        )
        if h_behavior.verdict is Verdict.DIVERGENT:
            return (
                Verdict.DIVERGENT,
                (
                    "deleting an edge in no triangle yields a divergent graph "
                    "by Theorem 6.2"
                ),
                Certificate(
                    rule="non_triangle_edge",
                    target=h,
                    target_status="proven_divergent",
                    target_label=h_behavior.certificate.target_label
                    if h_behavior.certificate is not None
                    else None,
                    edge=(u, v),
                ),
            )
        if (
            h_behavior.certificate is not None
            and h_behavior.certificate.target_status == "conjectured_divergent"
        ):
            return (
                Verdict.INDETERMINATE,
                (
                    "deleting an edge in no triangle yields a conjectured "
                    "divergent graph; conditional on the target being clique "
                    "divergent by Theorem 6.2"
                ),
                Certificate(
                    rule="non_triangle_edge",
                    target=h_behavior.certificate.target or h,
                    target_status="conjectured_divergent",
                    target_label=h_behavior.certificate.target_label,
                    edge=(u, v),
                ),
            )
    return None


def classify_non_triangle_edge(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
) -> ClassifierResult | None:
    """Classify clique behavior by deleting edges in no triangle.

    The decision uses Theorem 6.2.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries : int, optional
        Maximum number of iterates to inspect (default: 9).
    bound : int, optional
        Maximum number of cliques allowed at each iteration (default: 30).

    .. rubric:: Returns

    tuple[Verdict, str, Certificate | None] | None
        A tuple of ``(verdict, reason, certificate)`` if deleting an edge in no
        triangle yields a divergent (or conjectured divergent) graph; ``None``
        if this rule establishes no verdict.
    """
    return _classify_non_triangle_edge(graph, tries=tries, bound=bound)


def _classify_inverse_cutpoint_extension(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
    max_candidates: int = 64,
) -> ClassifierResult | None:
    """Classify by an inverse cutpoint extension, or return ``None``.

    Searches for a local cutpoint whose admissible split (see
    :mod:`pycliques.cutpoints`) followed by removing the newly added edge
    (Theorem 6.2) yields a graph with known clique behavior. The admissible
    split itself is justified by Theorem 6.1 applied to the constructed
    edge, not by any additional unverified hypothesis.

    ``extension_budget`` bounds how many times this (expensive) rule may
    fire along a single recursive classification chain; it is decremented
    only when this rule itself is applied, not by the cheaper Theorem 6.1 /
    6.2 rules. A budget of 0 makes this rule return ``None`` immediately.
    """
    if extension_budget <= 0:
        return None

    for extension in inverse_cutpoint_extensions(graph, max_candidates=max_candidates):
        if not is_admissible_inverse_extension(extension):
            continue
        u, v = extension.u, extension.v
        extended = extension.graph
        if edge_in_triangle(extended, u, v):
            continue
        target = remove_edge_not_in_triangle(extended, u, v)
        target_behavior = classify_clique_behavior(
            target,
            tries=tries,
            bound=bound,
            extension_budget=extension_budget - 1,
        )
        certificate_map = {
            "cutpoint": extension.cutpoint,
            "u": extension.u,
            "v": extension.v,
        }
        if target_behavior.verdict is Verdict.DIVERGENT:
            return (
                Verdict.DIVERGENT,
                (
                    "an inverse cutpoint extension (Theorem 6.1) followed by "
                    "deleting the added edge, contained in no triangle "
                    "(Theorem 6.2), yields a divergent graph"
                ),
                Certificate(
                    rule="inverse_cutpoint_extension",
                    target=target,
                    target_status="proven_divergent",
                    target_label=target_behavior.certificate.target_label
                    if target_behavior.certificate is not None
                    else None,
                    map=certificate_map,
                    edge=(u, v),
                ),
            )
        if (
            target_behavior.certificate is not None
            and target_behavior.certificate.target_status == "conjectured_divergent"
        ):
            return (
                Verdict.INDETERMINATE,
                (
                    "an inverse cutpoint extension (Theorem 6.1) followed by "
                    "deleting the added edge (Theorem 6.2) yields a "
                    "conjectured divergent graph; conditional on the target "
                    "being clique divergent"
                ),
                Certificate(
                    rule="inverse_cutpoint_extension",
                    target=target_behavior.certificate.target or target,
                    target_status="conjectured_divergent",
                    target_label=target_behavior.certificate.target_label,
                    map=certificate_map,
                    edge=(u, v),
                ),
            )
    return None


def classify_inverse_cutpoint_extension(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
    max_candidates: int = 64,
) -> ClassifierResult | None:
    """Classify clique behavior by an inverse cutpoint extension.

    This searches for a local cutpoint of ``graph`` that admits a
    theorem-valid split into two vertices ``u``, ``v`` joined by a new edge
    (Theorem 6.1), such that deleting that edge -- contained in no triangle
    by construction requirements -- yields a graph of known clique behavior
    (Theorem 6.2). See :mod:`pycliques.cutpoints` for the precise
    admissibility condition.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries : int, optional
        Maximum number of iterates to inspect (default: 9).
    bound : int, optional
        Maximum number of cliques allowed at each iteration (default: 30).
    extension_budget : int, optional
        Maximum number of times this rule may recursively fire along one
        classification chain (default: 1).
    max_candidates : int, optional
        Maximum number of split candidates generated per local cutpoint
        (default: 64).

    .. rubric:: Returns

    tuple[Verdict, str, Certificate | None] | None
        A tuple of ``(verdict, reason, certificate)`` if an admissible
        extension leads to a graph of known clique behavior; ``None`` if
        this rule establishes no verdict.
    """
    return _classify_inverse_cutpoint_extension(
        graph,
        tries=tries,
        bound=bound,
        extension_budget=extension_budget,
        max_candidates=max_candidates,
    )


def _clockwork_source_order(m: int, n: int) -> int:
    """Return the order of R_{2m}^n without constructing the graph."""
    return 4 * m + 2 * m * (n + 1)


def _find_clockwork_pair_certificate(
    target_graph: nx.Graph,
    *,
    max_m: int,
    max_n: int,
    max_coaffinations: int,
    max_source_order: int,
) -> tuple[int, int, int, CoaffinePair, dict, dict] | None:
    """Search for a Theorem 2.6 clockwork pair map into *target_graph*.

    Tries every ``m`` in ``2 .. max_m``; for each, collects up to
    ``max_coaffinations`` involutive ``(m + 1)``-coaffinations of
    ``target_graph``, then tries every ``n`` in ``0 .. max_n`` (subject to
    ``max_source_order``) against each candidate coaffination.

    ``m == 1`` (radius 2) is never tried: Theorem 3.1 only certifies that
    ``R_{2m}^n`` is rank divergent for ``m >= 2``, so a 1- or 2-coaffination
    target would not be backed by that theorem's hypothesis.

    .. rubric:: Returns

    tuple[int, int, int, CoaffinePair, dict, dict] | None
        ``(m, n, radius, source_pair, tau, f)`` for the first admissible
        map found, or ``None`` if none exists within the configured
        bounds.
    """
    for m in range(2, max_m + 1):
        radius = m + 1
        candidate_taus = []
        for tau in candidate_target_coaffinations(target_graph, radius):
            candidate_taus.append(tau)
            if len(candidate_taus) >= max_coaffinations:
                break
        if not candidate_taus:
            continue
        for n in range(0, max_n + 1):
            if _clockwork_source_order(m, n) > max_source_order:
                break
            source_pair = clockwork_coaffine_pair(m, n)
            for tau in candidate_taus:
                target_pair = CoaffinePair(target_graph, tau)
                f = find_pair_morphism(source_pair, target_pair)
                if f is not None:
                    return (m, n, radius, source_pair, tau, f)
    return None


def _classify_clockwork_pair_map(
    graph: nx.Graph,
    *,
    max_m: int = 3,
    max_n: int = 3,
    max_coaffinations: int = 20,
    max_source_order: int = 40,
    bound: int = 30,
) -> ClassifierResult | None:
    """Classify by Theorem 2.6 applied to Theorem 3.1, or return ``None``.

    Searches for an admissible graph morphism of coaffine pairs
    ``f: (R_{2m}^n, sigma) -> (graph, tau)`` for some ``m`` in
    ``2 .. max_m``, ``n`` in ``0 .. max_n``, and involutive
    ``(m + 1)``-coaffination ``tau`` of ``graph``.  By Theorem 3.1,
    ``R_{2m}^n`` is ``(m + 1)``-coaffine and rank divergent for ``m >= 2``;
    by Theorem 2.6, an admissible morphism to ``(graph, tau)`` transfers
    rank divergence to ``graph``, and rank divergence implies clique
    divergence.  ``m == 1`` (radius 2) is never searched: Theorem 3.1 does
    not certify rank divergence of ``R_2^n``, so a 1- or 2-coaffination
    target would not be a sound certificate.

    If this direct search fails, ``K(graph)`` is computed (respecting
    ``bound``, exactly as :class:`CliqueSequence` would) and the same
    bounded search is retried with ``K(graph)`` as the target.  A
    successful map into ``K(graph)`` still certifies divergence of
    ``graph``: rank divergence of an iterated clique graph implies rank
    divergence -- and hence clique divergence -- of the original graph.
    This is essential for examples such as the icosahedron, whose own
    clockwork pair map search fails but whose clique graph admits one.

    A failed search establishes nothing: it is not evidence that ``graph``
    is convergent, nor that it lacks a suitable coaffination.  If
    ``K(graph)`` cannot be computed within ``bound``, this rule returns
    ``None`` rather than treating that as a negative result.

    ``max_coaffinations`` bounds, for each ``m``, the number of candidate
    target coaffinations examined (shared across every ``n`` tried for that
    ``m``).  ``max_source_order`` stops increasing ``n`` once ``R_{2m}^n``
    would exceed that many vertices.  These bounds apply identically to the
    direct search and to the ``K(graph)`` search.
    """
    direct = _find_clockwork_pair_certificate(
        graph,
        max_m=max_m,
        max_n=max_n,
        max_coaffinations=max_coaffinations,
        max_source_order=max_source_order,
    )
    if direct is not None:
        m, n, radius, source_pair, tau, f = direct
        return (
            Verdict.DIVERGENT,
            (
                f"admits a map of coaffine pairs from R_{{{2 * m}}}^"
                f"{{{n}}} (Theorem 3.1) yielding rank divergence "
                "transferred by Theorem 2.6"
            ),
            Certificate(
                rule="theorem_2_6_clockwork_pair_map",
                target=graph,
                target_status="proven_divergent",
                map=f,
                m=m,
                n=n,
                radius=radius,
                source=source_pair.graph,
                source_coaffination=source_pair.coaffination,
                target_coaffination=tau,
            ),
        )

    kg = clique_graph(graph, bound)
    if kg is None:
        return None

    via_clique_graph = _find_clockwork_pair_certificate(
        kg,
        max_m=max_m,
        max_n=max_n,
        max_coaffinations=max_coaffinations,
        max_source_order=max_source_order,
    )
    if via_clique_graph is None:
        return None
    m, n, radius, source_pair, tau, f = via_clique_graph
    return (
        Verdict.DIVERGENT,
        (
            f"admits a map of coaffine pairs from R_{{{2 * m}}}^{{{n}}} "
            "(Theorem 3.1) into K(graph), yielding rank divergence of "
            "K(graph) by Theorem 2.6, hence rank divergence -- and clique "
            "divergence -- of the original graph"
        ),
        Certificate(
            rule="theorem_2_6_clockwork_pair_map_clique_graph",
            target=kg,
            target_status="proven_divergent",
            map=f,
            m=m,
            n=n,
            radius=radius,
            source=source_pair.graph,
            source_coaffination=source_pair.coaffination,
            target_coaffination=tau,
        ),
    )


def classify_clockwork_pair_map(
    graph: nx.Graph,
    *,
    max_m: int = 3,
    max_n: int = 3,
    max_coaffinations: int = 20,
    max_source_order: int = 40,
    bound: int = 30,
) -> ClassifierResult | None:
    """Classify clique behavior by a map of coaffine pairs (Theorems 2.6, 3.1).

    Searches for an integer ``m``, an integer ``n``, an involutive
    ``(m + 1)``-coaffination ``tau`` of ``graph``, and an admissible graph
    morphism ``f: (R_{2m}^n, sigma) -> (graph, tau)`` where ``sigma`` is the
    canonical coaffination of the clockwork graph :math:`R_{2m}^n`
    (:func:`pycliques.clockwork_pairs.clockwork_coaffine_pair`).

    :math:`R_{2m}^n` is :math:`(m+1)`-coaffine and rank divergent by Theorem
    3.1 for :math:`m \geq 2`.  An admissible morphism of coaffine pairs
    transfers rank divergence from the domain to the codomain by Theorem
    2.6, and rank divergence implies clique divergence.  ``m == 1`` (radius
    2) is never searched, since Theorem 3.1 does not certify rank
    divergence of :math:`R_2^n`.

    When the direct search fails, the same bounded search is retried with
    ``K(graph)`` as the target (computed subject to ``bound``): a map into
    ``K(graph)`` still certifies that ``graph`` is rank -- and hence clique
    -- divergent, since rank divergence of an iterated clique graph implies
    rank divergence of the original graph.  This second attempt is what
    lets this rule classify e.g. ``networkx.icosahedral_graph()``, whose
    own clockwork pair map search fails while its clique graph's succeeds.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    max_m : int, optional
        Maximum clockwork parameter ``m`` to try (default: 3).  Values of
        ``m`` below 2 (radius below 3) are never tried, regardless of this
        bound, since Theorem 3.1 only certifies rank divergence of
        ``R_{2m}^n`` for ``m >= 2``.
    max_n : int, optional
        Maximum clockwork parameter ``n`` to try (default: 3).
    max_coaffinations : int, optional
        Maximum number of candidate target coaffinations examined per ``m``
        (default: 20).
    max_source_order : int, optional
        Maximum order of ``R_{2m}^n`` to construct (default: 40).
    bound : int, optional
        Maximum number of cliques allowed when computing ``K(graph)``
        (default: 30).  If exceeded, this rule returns ``None`` instead of
        attempting the ``K(graph)`` search.

    .. rubric:: Returns

    tuple[Verdict, str, Certificate | None] | None
        A tuple of ``(Verdict.DIVERGENT, reason, certificate)`` if an
        admissible map of coaffine pairs is found -- into ``graph`` itself
        or into ``K(graph)`` -- within the configured bounds; ``None`` if
        this rule establishes no verdict.  This rule never returns
        ``Verdict.CONVERGENT``.
    """
    return _classify_clockwork_pair_map(
        graph,
        max_m=max_m,
        max_n=max_n,
        max_coaffinations=max_coaffinations,
        max_source_order=max_source_order,
        bound=bound,
    )


def _test_eventually_helly(seq: CliqueSequence, tries: int) -> ClassifierResult | None:
    """Return convergence if some iterate is clique-Helly, else ``None``."""
    for i in range(tries):
        g = seq[i]
        if g is None:
            return None
        if is_clique_helly(g):
            _logger.debug(f"Helly of index {i}")
            return (Verdict.CONVERGENT, f"is eventually Helly (index {i})", None)
    return None


def _test_clockwork(seq: CliqueSequence, tries: int) -> ClassifierResult | None:
    """Return a clockwork verdict, or ``None`` if this rule proves nothing."""
    for i in range(min(2, tries)):
        g = seq[i]
        if g is None:
            return None
        if recognize_clockwork(g)[0]:
            divergent, _ = is_clique_divergent_clockwork(g)
            if divergent is True:
                return (Verdict.DIVERGENT, "is clockwork divergent", None)
            if divergent is False:
                return (Verdict.CONVERGENT, "is clockwork convergent", None)
    return None


def _test_eventually_special_octahedra(
    seq: CliqueSequence, tries: int
) -> ClassifierResult | None:
    """Return divergence if found, or ``None`` if this rule proves nothing."""
    for i in range(tries):
        g = seq[i]
        if g is None:
            return None
        dim = special_octahedra_dimension(g)
        if dim is not None:
            _logger.debug(f"Index {i} has induced special octahedra")
            return (
                Verdict.DIVERGENT,
                f"eventually has a special octahedron (index {i}, dimension {dim})",
                None,
            )
    return None


def _make_retraction_test(target: nx.Graph, label: str) -> Classifier:
    """Return a classifier that checks whether ``seq[0]`` retracts to *target*."""

    def _test(seq: CliqueSequence) -> ClassifierResult | None:
        g = seq[0]
        if g is not None and retracts(g, target):
            return (Verdict.DIVERGENT, label, None)
        return None

    return _test


def _make_clique_retraction_test(target: nx.Graph, label: str) -> Classifier:
    """Return a classifier that checks whether ``seq[1]`` retracts to *target*."""

    def _test(seq: CliqueSequence) -> ClassifierResult | None:
        g = seq[1]
        if g is not None and retracts(g, target):
            return (Verdict.DIVERGENT, label, None)
        return None

    return _test


def _default_classifiers(tries: int) -> list[Classifier]:
    """Return the standard ordered suite of clique-behavior tests."""
    classifiers = [
        lambda seq: _test_clockwork(seq, tries),
        lambda seq: _test_eventually_helly(seq, tries),
        lambda seq: _test_eventually_special_octahedra(seq, tries),
        _make_retraction_test(suspension_of_cycle(5), "retracts to Susp(C_5)"),
        _make_retraction_test(suspension_of_cycle(6), "retracts to Susp(C_6)"),
        _make_retraction_test(suspension_of_cycle(7), "retracts to Susp(C_7)"),
        _make_retraction_test(complement_of_cycle(8), "retracts to Comp(C_8)"),
    ]
    return classifiers


def classify_clique_behavior(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
) -> CliqueBehavior:
    """Classify the observed clique behavior of an undirected graph.

    This is the *fast pass*: it deliberately excludes the expensive
    Theorem 2.6 / coaffination search
    (:func:`classify_clockwork_pair_map`).  A graph left ``INDETERMINATE``
    by this fast pass may still be classifiable by that separate, more
    expensive rule; see :func:`classify_clique_behavior_with_theorem_2_6`
    to run both stages together.

    The input is completely pared before its iterated clique graphs are
    inspected.  The standard test suite certifies convergence through an
    eventually clique-Helly iterate, and divergence through clockwork,
    special-octahedron, and known-divergence criteria.

    The classifier applies a sequence of mathematically justified
    inference rules. These include retraction to registered reference
    graphs, Theorem 6.1 local-bridge contractions, Theorem 6.2
    deletion of edges contained in no triangle, and theorem-motivated
    inverse cutpoint extensions. Each rule returns None when it
    establishes no verdict, allowing the pipeline to continue. A
    conjecturally divergent reference produces a conditional
    INDETERMINATE result rather than a proof of divergence.

    Each inference rule returns ``None`` when it establishes no verdict, so
    the pipeline continues.  The final ``INDETERMINATE`` result means that
    the available (fast) rules were exhausted without a definitive
    conclusion -- it is not evidence about what the (unrun) Theorem 2.6
    search would find.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries : int, optional
        Maximum number of iterates to inspect, including the input
        (default: 9).
    bound : int, optional
        Maximum number of cliques allowed at each iteration
        (default: 30).

    .. rubric:: Returns

    CliqueBehavior
        A certified verdict when a sufficient test succeeds; otherwise an
        indeterminate result.  ``bound_exceeded`` distinguishes an aborted
        clique-graph computation from exhaustion of ``tries``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.small import Verdict, classify_clique_behavior
    >>> result = classify_clique_behavior(nx.cycle_graph(4))
    >>> result.verdict is Verdict.CONVERGENT
    True

    """
    if tries < 1:
        raise ValueError("tries must be at least 1")
    if bound < 1:
        raise ValueError("bound must be at least 1")

    pared_graph = completely_pared_graph(graph)
    seq = CliqueSequence(pared_graph, bound=bound, on_iterate=on_iterate)

    reference_result = _classify_reference_graph(pared_graph)
    if reference_result is not None:
        verdict, reason, certificate = reference_result
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    theorem_4_6_result = _classify_suspension_2_coaffination(graph)
    if theorem_4_6_result is not None:
        verdict, reason, certificate = theorem_4_6_result
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    for classifier in _default_classifiers(tries):
        result = classifier(seq)
        if result is not None:
            verdict, reason, certificate = result
            return CliqueBehavior(
                verdict, reason, seq.graph_count, False, pared_graph, certificate
            )

    reference_dependency = _classify_reference_dependency(pared_graph)
    if reference_dependency is not None:
        verdict, reason, certificate = reference_dependency
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    local_bridge_result = _classify_local_bridge(
        pared_graph, tries=tries, bound=bound, extension_budget=extension_budget
    )
    if local_bridge_result is not None:
        verdict, reason, certificate = local_bridge_result
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    non_triangle_edge_result = _classify_non_triangle_edge(
        pared_graph, tries=tries, bound=bound, extension_budget=extension_budget
    )
    if non_triangle_edge_result is not None:
        verdict, reason, certificate = non_triangle_edge_result
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    inverse_extension_result = _classify_inverse_cutpoint_extension(
        pared_graph, tries=tries, bound=bound, extension_budget=extension_budget
    )
    if inverse_extension_result is not None:
        verdict, reason, certificate = inverse_extension_result
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    bound_exceeded = seq.exhausted
    reason = (
        "clique count exceeded bound"
        if bound_exceeded
        else "behavior is indeterminate under the available tests"
    )
    return CliqueBehavior(
        Verdict.INDETERMINATE,
        reason,
        seq.graph_count,
        bound_exceeded,
        pared_graph,
    )


def classify_clique_behavior_with_theorem_2_6(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
    max_m: int = 3,
    max_n: int = 3,
    max_coaffinations: int = 20,
    max_source_order: int = 40,
    theorem_2_6_bound: int | None = None,
) -> CliqueBehavior:
    """Run the fast pass, then the expensive Theorem 2.6 pass if still unresolved.

    This is the two-stage workflow used by the ``small-behavior-theorem26``
    CLI: :func:`classify_clique_behavior` (the fast pass) is run first, and
    :func:`classify_clockwork_pair_map` (the Theorem 2.6 / coaffination
    search) is only attempted when the fast pass leaves the graph
    ``INDETERMINATE``.  Graphs already classified as ``CONVERGENT`` or
    ``DIVERGENT`` never reach the expensive search.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries, bound, extension_budget
        Forwarded to :func:`classify_clique_behavior`.
    max_m, max_n, max_coaffinations, max_source_order
        Forwarded to :func:`classify_clockwork_pair_map`.
    theorem_2_6_bound : int, optional
        Clique-graph bound used by the Theorem 2.6 search's ``K(graph)``
        fallback.  Defaults to ``bound`` when not given.

    .. rubric:: Returns

    CliqueBehavior
        The fast-pass result unchanged, unless the fast pass was
        ``INDETERMINATE`` and Theorem 2.6 establishes ``DIVERGENT``.
    """
    fast = classify_clique_behavior(
        graph, tries=tries, bound=bound, extension_budget=extension_budget
    )
    if fast.verdict is not Verdict.INDETERMINATE:
        return fast

    theorem_bound = bound if theorem_2_6_bound is None else theorem_2_6_bound
    theorem_result = classify_clockwork_pair_map(
        fast.pared_graph,
        max_m=max_m,
        max_n=max_n,
        max_coaffinations=max_coaffinations,
        max_source_order=max_source_order,
        bound=theorem_bound,
    )
    if theorem_result is None:
        return fast
    verdict, reason, certificate = theorem_result
    return CliqueBehavior(
        verdict,
        reason,
        fast.iterations_checked,
        fast.bound_exceeded,
        fast.pared_graph,
        certificate,
    )


_DEFAULT_DATA_DIR = Path(".")


def _indeterminate_file_path(order: int, data_dir: Path) -> Path:
    """Return the path for the indeterminate-graphs file of a given order."""
    return data_dir / f"indeterminate_order_{order}.txt"


def _serialize_certificate(certificate: Certificate | None) -> str:
    """Render certificate metadata for the indeterminate-graphs file."""
    if certificate is None:
        return "- - -"
    target_label = certificate.target_label or "-"
    target_status = certificate.target_status or "-"
    return f"{certificate.rule} {target_status} {target_label}"


def _save_indeterminate(
    order: int,
    indeterminate: list[tuple[int, nx.Graph, Certificate | None]],
    data_dir: Path,
    *,
    replace: bool = False,
) -> None:
    """Save indeterminate pared graphs to a human-readable file.

    Each line contains the original graph index, the order of the pared
    graph, its graph6 string, and certificate metadata when available.

    If the file already exists, the new results are merged with the
    existing entries.  Duplicate indices are resolved in favour of the
    new run so that re-processing a range always updates the record.  When
    ``replace`` is true, the file is replaced by exactly the supplied
    indeterminate entries.
    """
    path = _indeterminate_file_path(order, data_dir)

    # Load existing entries keyed by original index.
    existing: dict[int, str] = {}
    if path.is_file() and not replace:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split(maxsplit=5)
                existing[int(parts[0])] = stripped
        _logger.info(
            f"Found existing file with {len(existing)} entries; merging new results"
        )

    # Build lines for the new entries (overwrite any duplicate index).
    for idx, graph, certificate in indeterminate:
        g = nx.convert_node_labels_to_integers(graph)
        g6 = nx.to_graph6_bytes(g, header=False).decode("ascii").strip()
        existing[idx] = f"{idx} {g.order()} {g6} {_serialize_certificate(certificate)}"

    # Write everything back sorted by index.
    with path.open("w", encoding="utf-8") as f:
        f.write(
            f"# Indeterminate clique behavior - connected graphs of order {order}\n"
        )
        f.write(
            "# Format: original_index pared_order graph6 certificate_rule "
            "certificate_target_status certificate_target_label\n"
        )
        for idx in sorted(existing):
            f.write(f"{existing[idx]}\n")
    _logger.info(f"Saved {len(existing)} indeterminate graphs to {path}")


def _graph6_string(graph: nx.Graph) -> str:
    """Return a graph6 string after normalizing arbitrary node labels."""
    numbered = nx.convert_node_labels_to_integers(graph)
    return cast(
        str, nx.to_graph6_bytes(numbered, header=False).decode("ascii").strip()
    )


def _recheck_indeterminate_file(
    entries: list[tuple[int, nx.Graph, Certificate | None]],
    bound: int,
    classifier: Classifier | None = None,
) -> list[tuple[int, nx.Graph, Certificate | None]]:
    """Re-run a classifier on graphs saved as indeterminate."""
    still_indeterminate: list[tuple[int, nx.Graph, Certificate | None]] = []
    for index, graph, saved_certificate in entries:
        _logger.info("Studying graph %s (%s)", index, _graph6_string(graph))
        if classifier is None:
            behavior = classify_clique_behavior(graph, bound=bound)
            if behavior.verdict is Verdict.INDETERMINATE:
                still_indeterminate.append(
                    (
                        index,
                        behavior.pared_graph,
                        behavior.certificate or saved_certificate,
                    )
                )
            else:
                _logger.info(
                    "Graph %s is now %s: %s",
                    index,
                    behavior.verdict.name,
                    behavior.reason,
                )
        else:
            sequence = CliqueSequence(graph, bound=bound)
            rule_result = classifier(sequence)
            if rule_result is None:
                still_indeterminate.append((index, graph, saved_certificate))
            else:
                verdict, reason, certificate = rule_result
                _logger.info("Graph %s is now %s: %s", index, verdict.name, reason)

    return still_indeterminate


def _load_indeterminate_graphs(
    max_order: int,
    data_dir: Path,
) -> dict[int, list[nx.Graph]]:
    """Load indeterminate graphs from files for all orders less than *max_order*.

    .. rubric:: Returns

    dict[int, list[nx.Graph]]
        Graphs grouped by their vertex count so that lookups only need to
        test isomorphism against candidates of the same size.
    """
    by_vertex_count: dict[int, list[nx.Graph]] = {}
    for order in range(1, max_order):
        path = _indeterminate_file_path(order, data_dir)
        if not path.is_file():
            continue
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=5)
                g6_str = parts[2]
                graph = nx.from_graph6_bytes(g6_str.encode("ascii"))
                n = graph.order()
                by_vertex_count.setdefault(n, []).append(graph)
                count += 1
        _logger.info(f"Loaded {count} indeterminate graphs from order-{order} file")
    return by_vertex_count


def _is_known_indeterminate(
    graph: nx.Graph,
    known: dict[int, list[nx.Graph]],
) -> bool:
    """Return whether *graph* is isomorphic to any known indeterminate graph."""
    candidates = known.get(graph.order(), [])
    return any(nx.is_isomorphic(graph, c) for c in candidates)


def test_eventually_helly(
    graph: nx.Graph, tries: int = 8, bound: int = 30
) -> ClassifierResult:
    """Test whether a finite range of iterates contains a clique-Helly graph.

    Finding a clique-Helly iterate establishes eventual clique-Helly behavior.
    Failure to find one in the finite tested range is inconclusive and returns
    ``Verdict.INDETERMINATE``.

    Starting from ``graph``, repeatedly compute the completely-pared clique
    graph. Return a convergent result as soon as one iterate is clique-Helly.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries : int
        Maximum number of clique-graph iterations (default 8).
    bound : int
        Maximum number of cliques allowed before aborting (default 30).

    .. rubric:: Returns

    ClassifierResult
        A convergent result if an iterated clique graph within ``tries`` steps
        is clique-Helly. Otherwise, an indeterminate result; the reason
        identifies bound exhaustion when applicable.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.helly import is_clique_helly
    >>> from pycliques.small import Verdict, test_eventually_helly
    >>> is_clique_helly(nx.triangular_lattice_graph(3,3))
    False
    >>> test_eventually_helly(nx.triangular_lattice_graph(3,3))[0] is Verdict.CONVERGENT
    True

    """
    seq = CliqueSequence(graph, bound=bound)
    result = _test_eventually_helly(seq, tries + 1)
    if result is not None:
        return result
    if seq.exhausted:
        return (
            Verdict.INDETERMINATE,
            "clique count exceeded bound before finding a clique-Helly iterate",
            None,
        )
    return (
        Verdict.INDETERMINATE,
        "no clique-Helly iterate found in the finite tested range",
        None,
    )


def eventually_retracts_specially(
    graph: nx.Graph, tries: int = 8, bound: int = 20
) -> bool | None:
    """Check if iterated clique graphs eventually contain a special octahedron.

    Starting from ``graph``, repeatedly compute the completely-pared clique
    graph. Return ``True`` as soon as one iterate contains a special
    octahedron (see :func:`pycliques.retractions.special_octahedra`).

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries : int
        Maximum number of clique-graph iterations (default 8).
    bound : int
        Maximum number of cliques before aborting (default 20).

    .. rubric:: Returns

    bool | None
        ``True`` if a special octahedron is found, ``None`` if the
        computation is inconclusive.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.small import eventually_retracts_specially
    >>> from pyg6data.lists import list_graphs
    >>> g = list_graphs(8)[11045]
    >>> eventually_retracts_specially(g)
    True

    """
    seq = CliqueSequence(graph, bound=bound)
    result = _test_eventually_special_octahedra(seq, tries)
    return True if result is not None else None


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the small-graphs script."""
    parser = argparse.ArgumentParser(description="Clique behavior of small graphs")
    parser.add_argument(
        "--version", action="version", version=f"pycliques {__version__}"
    )
    parser.add_argument(
        dest="n", help="Order of graphs to consider (e.g., 6)", type=int, metavar="INT"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="Set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        help="Do not save indeterminate graphs to file",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--no-lookup",
        dest="lookup",
        help="Do not look up pared graphs in prior indeterminate files",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--from-indeterminate-file",
        dest="from_indeterminate_file",
        help=(
            "Read graphs from indeterminate_order_<n>.txt, re-run the default "
            "tests, and remove graphs that are resolved"
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--exclude-conjectured-divergent",
        dest="exclude_conjectured_divergent",
        help=(
            "In --from-indeterminate-file mode, do not re-run graphs whose "
            "saved certificate marks them conjectured divergent"
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--check-clique-retraction",
        dest="check_clique_retraction",
        help=(
            "In --from-indeterminate-file mode, run the seq[1] retraction "
            "test to complement(C_10)"
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        help="Directory for indeterminate graph files (default: current directory)",
        type=Path,
        default=_DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--bound",
        dest="bound",
        help="Maximum number of cliques allowed at each iteration (default: 30)",
        type=int,
        default=30,
        metavar="INT",
    )
    parser.add_argument(
        "--start",
        dest="start",
        help="First graph index to process, inclusive (default: 0)",
        type=int,
        default=None,
        metavar="INT",
    )
    parser.add_argument(
        "--end",
        dest="end",
        help="Last graph index to process, inclusive (default: last graph)",
        type=int,
        default=None,
        metavar="INT",
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        help=(
            "Write one verdict line per graph to this file. "
            "Format: index TAB verdict TAB reason"
        ),
        type=Path,
        default=None,
        metavar="FILE",
    )
    parser.add_argument(
        "--skip-dominated",
        dest="skip_dominated",
        help="Skip graphs that have dominated vertices (default: True)",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(args)


def _setup_logging(loglevel: int):
    """Configure Rich logging for the CLI entry point."""
    logging.basicConfig(
        level=loglevel, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )


def _main(args: list[str]):
    """Run the small-graph classification from parsed CLI arguments."""
    from pyg6data.lists import _dict_connected, _get_data_file_path

    parsed_args = _parse_args(args)
    _setup_logging(parsed_args.loglevel)

    order = parsed_args.n
    save = parsed_args.save
    lookup = parsed_args.lookup
    data_dir: Path = parsed_args.data_dir
    start: int | None = parsed_args.start
    end: int | None = parsed_args.end
    output_file: Path | None = parsed_args.output_file
    skip_dominated: bool = parsed_args.skip_dominated
    bound: int = parsed_args.bound

    if order not in _dict_connected:
        _logger.error(f"Error: Internal data for order {order} not available.")
        sys.exit(1)

    if start is not None and start < 0:
        _logger.error("--start must be a non-negative integer.")
        sys.exit(1)

    if end is not None and end < 0:
        _logger.error("--end must be a non-negative integer.")
        sys.exit(1)

    if start is not None and end is not None and start > end:
        _logger.error("--start must be less than or equal to --end.")
        sys.exit(1)

    if parsed_args.from_indeterminate_file:
        entries_with_metadata = _load_indeterminate_file_entries_with_metadata(
            order, data_dir
        )
        entries = [
            (index, graph, certificate)
            for index, graph, certificate in entries_with_metadata
            if not (
                parsed_args.exclude_conjectured_divergent
                and certificate is not None
                and certificate.target_status == "conjectured_divergent"
            )
        ]
        excluded: list[tuple[int, nx.Graph, Certificate | None]] = [
            (index, graph, certificate)
            for index, graph, certificate in entries_with_metadata
            if parsed_args.exclude_conjectured_divergent
            and certificate is not None
            and certificate.target_status == "conjectured_divergent"
        ]
        _logger.info(
            "Rechecking %s graphs from %s...",
            len(entries),
            _indeterminate_file_path(order, data_dir),
        )
        recheck_classifier = None
        if parsed_args.check_clique_retraction:
            recheck_classifier = _make_clique_retraction_test(
                complement_of_cycle(10),
                "clique graph retracts to Comp(C_10)",
            )
        still_indeterminate = _recheck_indeterminate_file(
            entries, bound, classifier=recheck_classifier
        )
        if save:
            _save_indeterminate(
                order,
                still_indeterminate + excluded,
                data_dir,
                replace=True,
            )
        _logger.info(
            "Recheck complete: %s remain indeterminate, %s resolved.",
            len(still_indeterminate),
            len(entries) - len(still_indeterminate),
        )
        return

    _logger.info("Precomputing target mathematical structures...")
    # Load previously saved indeterminate graphs for smaller orders
    known_indeterminate: dict[int, list[nx.Graph]] = {}
    if lookup:
        known_indeterminate = _load_indeterminate_graphs(order, data_dir)

    convergent: list[int] = []
    divergent: list[int] = []
    further: list[int] = []
    further_pared: list[tuple[int, nx.Graph, Certificate | None]] = []
    further_graphs: list[tuple[int, nx.Graph, Certificate | None]] = []
    reducible: list[int] = []

    range_msg = ""
    if start is not None or end is not None:
        lo = start if start is not None else 0
        hi = end if end is not None else "last"
        range_msg = f" (indices {lo}-{hi})"
    _logger.info(
        f"Beginning analysis of connected graphs of order {order}{range_msg}..."
    )

    # 2. Securely resolve the dataset path
    data_path = _get_data_file_path(_dict_connected[order])

    import contextlib

    output_ctx = (
        open(output_file, "w", encoding="utf-8")  # noqa: WPS515
        if output_file is not None
        else contextlib.nullcontext()
    )

    index = -1
    with output_ctx as verdict_file:
        if verdict_file is not None:
            verdict_file.write("# index\tverdict\treason\n")
            _logger.info(f"Writing verdicts to {output_file}")

        with data_path.open("rb") as raw_file:
            with gzip.open(raw_file, "rt", encoding="utf-8") as graph_file:
                for index, line in enumerate(graph_file):
                    if start is not None and index < start:
                        continue
                    if end is not None and index > end:
                        break

                    if index > 0 and index % 10000 == 0:  # pragma: no cover
                        _logger.info(f"Processed up to index {index}...")

                    assert isinstance(line, str)
                    graph = nx.from_graph6_bytes(bytes(line.strip(), "utf-8"))

                    if skip_dominated and find_dominated_vertex(graph) is not None:
                        reducible.append(index)
                        if verdict_file is not None:
                            verdict_file.write(
                                f"{index}\tREDUCIBLE\thas dominated vertices\n"
                            )
                        _logger.debug(f"Graph {index}: has dominated vertices")
                        continue

                    result = classify_clique_behavior(graph, bound=bound)
                    if _is_known_indeterminate(result.pared_graph, known_indeterminate):
                        further_pared.append(
                            (index, result.pared_graph, result.certificate)
                        )
                        further_graphs.append(
                            (index, result.pared_graph, result.certificate)
                        )
                        verdict_label = "INDETERMINATE"
                        reason = "reduces to known indeterminate graph"
                    elif result.verdict is Verdict.CONVERGENT:
                        convergent.append(index)
                        verdict_label = "CONVERGENT"
                        reason = result.reason
                    elif result.verdict is Verdict.DIVERGENT:
                        divergent.append(index)
                        verdict_label = "DIVERGENT"
                        reason = result.reason
                    else:
                        further.append(index)
                        further_graphs.append(
                            (index, result.pared_graph, result.certificate)
                        )
                        verdict_label = "INDETERMINATE"
                        reason = result.reason

                    if verdict_file is not None:
                        verdict_file.write(f"{index}\t{verdict_label}\t{reason}\n")
                    _logger.debug(f"Graph {index}: {reason}")

    total_processed = len(convergent) + len(divergent) + len(further) + len(reducible)
    _logger.info(f"Analysis Complete! Processed {total_processed} total graphs.")
    _logger.info(f"Indices that deserve further study: {further}")
    _logger.info(f"Total convergent graphs: {len(convergent)}")
    _logger.info(f"Total divergent graphs: {len(divergent)}")
    _logger.info(f"Total reducible graphs skipped: {len(reducible)}")
    _logger.info(f"Total graphs reduced to indeterminate: {len(further_pared)}")
    _logger.info(f"Total indeterminate graphs (further study): {len(further)}")

    if save and further_graphs:
        _save_indeterminate(order, further_graphs, data_dir)


def main():  # pragma: no cover
    """Entry point for the ``clique-behavior`` console script."""
    _main(sys.argv[1:])


# ---------------------------------------------------------------------------
# Second-stage CLI: optional Theorem 2.6 pass over fast-pass INDETERMINATE
# graphs. See classify_clique_behavior_with_theorem_2_6 for the classifier
# orchestration; this section only handles CLI parsing, dataset iteration,
# and reporting.
# ---------------------------------------------------------------------------


def _load_indeterminate_file_entries(
    order: int, data_dir: Path
) -> list[tuple[int, nx.Graph]]:
    """Load ``(index, graph)`` pairs previously saved for *order* by the fast pass."""
    return [
        (index, graph)
        for index, graph, _certificate in (
            _load_indeterminate_file_entries_with_metadata(order, data_dir)
        )
    ]


def _load_indeterminate_file_entries_with_metadata(
    order: int, data_dir: Path
) -> list[tuple[int, nx.Graph, Certificate | None]]:
    """Load saved indeterminate entries together with certificate metadata."""
    path = _indeterminate_file_path(order, data_dir)
    entries: list[tuple[int, nx.Graph, Certificate | None]] = []
    if not path.is_file():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=5)
            certificate = None
            if len(parts) >= 6 and parts[3] != "-":
                certificate = Certificate(
                    rule=parts[3],
                    target_status=None if parts[4] == "-" else parts[4],
                    target_label=None if parts[5] == "-" else parts[5],
                )
            entries.append(
                (
                    int(parts[0]),
                    nx.from_graph6_bytes(parts[2].encode("ascii")),
                    certificate,
                )
            )
    return entries


def _run_fast_pass_indeterminate(
    order: int,
    *,
    data_dir: Path,
    start: int | None,
    end: int | None,
    skip_dominated: bool,
    bound: int,
) -> tuple[int, int, list[tuple[int, nx.Graph, Certificate | None]]]:
    """Run the fast pass over the dataset, returning only INDETERMINATE graphs.

    .. rubric:: Returns

    tuple[int, int, list[tuple[int, nx.Graph, Certificate | None]]]
        ``(convergent_count, divergent_count, indeterminate)`` where
        ``indeterminate`` holds ``(index, pared_graph, certificate)`` for
        every graph the fast pass left ``INDETERMINATE``.
    """
    from pyg6data.lists import _dict_connected, _get_data_file_path

    data_path = _get_data_file_path(_dict_connected[order])
    convergent = 0
    divergent = 0
    indeterminate: list[tuple[int, nx.Graph, Certificate | None]] = []

    with data_path.open("rb") as raw_file:
        with gzip.open(raw_file, "rt", encoding="utf-8") as graph_file:
            for index, line in enumerate(graph_file):
                if start is not None and index < start:
                    continue
                if end is not None and index > end:
                    break
                graph = nx.from_graph6_bytes(bytes(line.strip(), "utf-8"))
                if skip_dominated and find_dominated_vertex(graph) is not None:
                    continue
                result = classify_clique_behavior(graph, bound=bound)
                if result.verdict is Verdict.CONVERGENT:
                    convergent += 1
                elif result.verdict is Verdict.DIVERGENT:
                    divergent += 1
                else:
                    indeterminate.append(
                        (index, result.pared_graph, result.certificate)
                    )

    return convergent, divergent, indeterminate


def _parse_theorem26_args(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the Theorem 2.6 second-stage script."""
    parser = argparse.ArgumentParser(
        description=(
            "Theorem 2.6 (coaffination) pass over graphs the fast "
            "small-behavior pass left INDETERMINATE"
        )
    )
    parser.add_argument(
        "--version", action="version", version=f"pycliques {__version__}"
    )
    parser.add_argument(
        dest="n", help="Order of graphs to consider (e.g., 9)", type=int, metavar="INT"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="Set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        help="Directory for indeterminate graph files (default: current directory)",
        type=Path,
        default=_DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--from-indeterminate-file",
        dest="from_indeterminate_file",
        help=(
            "Read candidates from the indeterminate_order_<n> file saved by "
            "a previous fast pass instead of rerunning it"
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--start",
        dest="start",
        help="First graph index to process, inclusive (default: 0)",
        type=int,
        default=None,
        metavar="INT",
    )
    parser.add_argument(
        "--end",
        dest="end",
        help="Last graph index to process, inclusive (default: last graph)",
        type=int,
        default=None,
        metavar="INT",
    )
    parser.add_argument(
        "--skip-dominated",
        dest="skip_dominated",
        help="Skip graphs that have dominated vertices (default: True)",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fast-bound",
        dest="fast_bound",
        help="Clique bound used when rerunning the fast pass (default: 30)",
        type=int,
        default=30,
        metavar="INT",
    )
    parser.add_argument(
        "--max-m",
        dest="max_m",
        help="Maximum clockwork parameter m to try; radius = m + 1 (default: 3)",
        type=int,
        default=3,
        metavar="INT",
    )
    parser.add_argument(
        "--max-n",
        dest="max_n",
        help="Maximum clockwork parameter n to try (default: 3)",
        type=int,
        default=3,
        metavar="INT",
    )
    parser.add_argument(
        "--max-coaffinations",
        dest="max_coaffinations",
        help="Maximum candidate target coaffinations examined per m (default: 20)",
        type=int,
        default=20,
        metavar="INT",
    )
    parser.add_argument(
        "--max-source-order",
        dest="max_source_order",
        help="Maximum order of R_{2m}^n to construct (default: 40)",
        type=int,
        default=40,
        metavar="INT",
    )
    parser.add_argument(
        "--bound",
        dest="bound",
        help=(
            "Clique bound for the K(graph) fallback used by the Theorem 2.6 "
            "search (default: 30)"
        ),
        type=int,
        default=30,
        metavar="INT",
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        help=(
            "Write one verdict line per graph to this file. "
            "Format: index TAB verdict TAB reason"
        ),
        type=Path,
        default=None,
        metavar="FILE",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        help="Do not save the remaining indeterminate graphs to file",
        action="store_false",
        default=True,
    )
    return parser.parse_args(args)


def _main_theorem26(args: list[str]):
    """Run the Theorem 2.6 pass over fast-pass INDETERMINATE graphs."""
    parsed_args = _parse_theorem26_args(args)
    _setup_logging(parsed_args.loglevel)

    order = parsed_args.n
    data_dir: Path = parsed_args.data_dir

    if parsed_args.start is not None and parsed_args.start < 0:
        _logger.error("--start must be a non-negative integer.")
        sys.exit(1)
    if parsed_args.end is not None and parsed_args.end < 0:
        _logger.error("--end must be a non-negative integer.")
        sys.exit(1)
    if (
        parsed_args.start is not None
        and parsed_args.end is not None
        and parsed_args.start > parsed_args.end
    ):
        _logger.error("--start must be less than or equal to --end.")
        sys.exit(1)

    candidates: list[tuple[int, nx.Graph, Certificate | None]]
    if parsed_args.from_indeterminate_file:
        entries = _load_indeterminate_file_entries(order, data_dir)
        candidates = [(idx, graph, None) for idx, graph in entries]
        fast_convergent = fast_divergent = None
    else:
        from pyg6data.lists import _dict_connected

        if order not in _dict_connected:
            _logger.error(f"Error: Internal data for order {order} not available.")
            sys.exit(1)
        _logger.info("Running fast pass to collect INDETERMINATE graphs...")
        fast_convergent, fast_divergent, candidates = _run_fast_pass_indeterminate(
            order,
            data_dir=data_dir,
            start=parsed_args.start,
            end=parsed_args.end,
            skip_dominated=parsed_args.skip_dominated,
            bound=parsed_args.fast_bound,
        )

    _logger.info("Fast pass:")
    _logger.info(f"    convergent:    {fast_convergent}")
    _logger.info(f"    divergent:     {fast_divergent}")
    _logger.info(f"    indeterminate: {len(candidates)}")

    newly_divergent: list[tuple[int, nx.Graph, Certificate | None]] = []
    still_indeterminate: list[tuple[int, nx.Graph, Certificate | None]] = []

    import contextlib

    output_ctx = (
        open(parsed_args.output_file, "w", encoding="utf-8")  # noqa: WPS515
        if parsed_args.output_file is not None
        else contextlib.nullcontext()
    )

    with output_ctx as verdict_file:
        if verdict_file is not None:
            verdict_file.write("# index\tverdict\treason\n")

        for idx, graph, _fast_certificate in candidates:
            theorem_result = classify_clockwork_pair_map(
                graph,
                max_m=parsed_args.max_m,
                max_n=parsed_args.max_n,
                max_coaffinations=parsed_args.max_coaffinations,
                max_source_order=parsed_args.max_source_order,
                bound=parsed_args.bound,
            )
            if theorem_result is not None:
                _verdict, reason, certificate = theorem_result
                newly_divergent.append((idx, graph, certificate))
                if verdict_file is not None:
                    verdict_file.write(f"{idx}\tDIVERGENT\t{reason}\n")
            else:
                still_indeterminate.append((idx, graph, None))
                if verdict_file is not None:
                    verdict_file.write(
                        f"{idx}\tINDETERMINATE\tunresolved by Theorem 2.6\n"
                    )

    _logger.info("Theorem 2.6:")
    _logger.info(f"    newly divergent:         {len(newly_divergent)}")
    _logger.info(f"    remaining indeterminate: {len(still_indeterminate)}")

    if parsed_args.save and still_indeterminate:
        _save_indeterminate(order, still_indeterminate, data_dir)


def main_theorem26():  # pragma: no cover
    """Entry point for the ``small-behavior-theorem26`` console script."""
    _main_theorem26(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
