"""Determine the clique behavior of small graphs."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from itertools import combinations
from typing import cast

import networkx as nx

from pycliques.classification_types import Certificate as Certificate
from pycliques.classification_types import CliqueBehavior as CliqueBehavior
from pycliques.classification_types import ReferenceStatus as ReferenceStatus
from pycliques.classification_types import Verdict as Verdict
from pycliques.clique_sequence import CliqueSequence as CliqueSequence
from pycliques.cliques import (
    clique_graph,
)
from pycliques.clockwork import (
    ClockworkStructure,
    is_clique_divergent_clockwork,
    recognize_clockwork,
)
from pycliques.clockwork_pairs import (
    candidate_target_coaffinations,
    clockwork_coaffine_pair,
    find_pair_morphism,
)
from pycliques.clockwork_search import (
    CertificateVerificationResult,
    CertificateVerificationStatus,
    ClockworkSearchStatus,
    InducedClockworkWitness,
    find_divergent_clockwork_retract,
    find_induced_clockwork_subgraphs,
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
from pycliques.dominated import completely_pared_graph
from pycliques.helly import is_clique_helly
from pycliques.known_octahedral_witnesses import (
    snub_o24_matches_source,
    snub_o24_pairs_for_iterates,
)
from pycliques.named import complement_of_cycle, suspension_of_cycle
from pycliques.octahedral_retractions import find_octahedral_retraction
from pycliques.references import (
    _REFERENCE_GRAPHS as _REFERENCE_GRAPHS,
)
from pycliques.references import (
    ReferenceSelection,
    current_references,
    reference_scope,
)
from pycliques.references import (
    clear_reference_graphs as clear_reference_graphs,
)
from pycliques.references import (
    register_reference_graph as register_reference_graph,
)
from pycliques.references import (
    select_reference_graphs as select_reference_graphs,
)
from pycliques.retractions import (
    RetractionSearchStatus,
    find_special_octahedron_witness,
    retracts,
    special_octahedra_dimension,
    verify_retraction_witness,
)
from pycliques.targeted_retractions import (
    TargetSearchStatus,
    search_targeted_retractions,
    verify_targeted_retraction,
)

from ._graph_utils import same_labeled_graph as _same_labeled_graph
from .indeterminate_files import (  # noqa: F401 -- historical entry points
    _graph6_string,
    _indeterminate_file_path,
    _is_known_indeterminate,
    _load_indeterminate_file_entries,
    _load_indeterminate_file_entries_with_metadata,
    _load_indeterminate_graphs,
    _save_indeterminate,
    _serialize_certificate,
    _update_processed_indeterminate,
)
from .small_cli import (  # noqa: F401 -- historical entry points
    _main,
    _main_clockwork_pairs,
    _parse_args,
    _parse_clockwork_pair_args,
    _recheck_indeterminate_file,
    _run_fast_pass_indeterminate,
    _setup_logging,
    main,
    main_clockwork_pairs,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classifier rule types
# ---------------------------------------------------------------------------


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
    """Apply a special case of [LNP09, Theorem 4.6] to a suspension.

    The implemented criterion is for ``H * complement(K_2)`` with ``H`` having
    a 2-coaffination; it is an application of the connected-summand theorem,
    not its full statement.
    """
    for u, v, base in suspension_bases(graph):
        for tau in coaffinations(base, 2):
            return (
                Verdict.DIVERGENT,
                (
                    "[LNP09, Theorem 4.6]: graph is the suspension of a "
                    "connected graph "
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


# ---------------------------------------------------------------------------
# Classifier functions
# ---------------------------------------------------------------------------

_MAX_ITERATIONS = 9


def _reference_match(
    graph: nx.Graph,
) -> tuple[nx.Graph, ReferenceStatus, str | None] | None:
    """Return the first registered reference graph isomorphic to *graph*, if any."""
    for ref_graph, status, label in current_references(_REFERENCE_GRAPHS.values()):
        if (
            len(graph) == len(ref_graph)
            and graph.size() == ref_graph.size()
            and nx.is_isomorphic(graph, ref_graph)
        ):
            return ref_graph, status, label
    return None


def _reference_identifier(reference_graph: nx.Graph, label: str | None) -> str:
    """Return a whitespace-free identifier for a registered reference graph.

    Unnamed references use the graph6 encoding of the stored graph in its own
    vertex order; this is reproducible but not a canonical form.
    """
    if label:
        return label
    numbered = nx.convert_node_labels_to_integers(reference_graph)
    g6 = nx.to_graph6_bytes(numbered, header=False).decode("ascii").strip()
    return f"graph6:{g6}"


def _classify_reference_graph(
    graph: nx.Graph,
) -> ClassifierResult | None:
    """Classify a graph directly from a registered reference fact, if present."""
    match = _reference_match(graph)
    if match is None:
        return None
    ref_graph, status, label = match
    ident = _reference_identifier(ref_graph, label)
    prefix = f"pared graph is isomorphic to registered reference {ident!r}"
    if status is ReferenceStatus.PROVEN:
        return (
            Verdict.DIVERGENT,
            f"{prefix} (proven clique divergent)",
            Certificate(
                rule="reference",
                target=ref_graph,
                target_status="proven_divergent",
                target_label=ident,
            ),
        )
    return (
        Verdict.INDETERMINATE,
        f"{prefix} (clique divergence conjectured; result remains indeterminate)",
        Certificate(
            rule="reference",
            target=ref_graph,
            target_status="conjectured_divergent",
            target_label=ident,
        ),
    )


def _classify_reference_dependency(
    graph: nx.Graph,
) -> ClassifierResult | None:
    """Classify by a reference-dependent retraction, or return ``None``.

    ``None`` means this inference rule did not establish a verdict; it does
    not make the overall graph indeterminate.
    """
    conditional: ClassifierResult | None = None
    for ref_graph, status, label in current_references(_REFERENCE_GRAPHS.values()):
        retraction = retracts(graph, ref_graph)
        if isinstance(retraction, tuple):
            ident = _reference_identifier(ref_graph, label)
            prefix = f"retracts to registered reference {ident!r}"
            if status is ReferenceStatus.PROVEN:
                return (
                    Verdict.DIVERGENT,
                    f"{prefix} (proven clique divergent)",
                    Certificate(
                        rule="retracts",
                        target=ref_graph,
                        target_status="proven_divergent",
                        target_label=ident,
                        map=retraction,
                    ),
                )
            if conditional is None:
                conditional = (
                    Verdict.INDETERMINATE,
                    f"{prefix} (conjectured clique divergent; conclusion conditional)",
                    Certificate(
                        rule="retracts",
                        target=ref_graph,
                        target_status="conjectured_divergent",
                        target_label=ident,
                        map=retraction,
                    ),
                )
    return conditional


def _classify_local_bridge(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
    retraction_depth: int = 1,
) -> ClassifierResult | None:
    """Classify by [FANP13, Theorem 6.1], or return ``None`` when it proves nothing.

    A successful certificate records the contracted edge.  Failure to classify
    a contracted target falls through to the next inference rule.
    """
    for u, v in local_bridges(graph):
        h = contract_local_bridge(graph, u, v)
        h_behavior = classify_clique_behavior(
            h,
            tries=tries,
            bound=bound,
            extension_budget=extension_budget,
            retraction_depth=retraction_depth,
        )
        if h_behavior.verdict is Verdict.CONVERGENT:
            return (
                Verdict.CONVERGENT,
                "contracting a local bridge yields a convergent graph by "
                "[FANP13, Theorem 6.1]",
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
                "contracting a local bridge yields a divergent graph by "
                "[FANP13, Theorem 6.1]",
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
                    "by [FANP13, Theorem 6.1]"
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
    """Apply [FANP13, Theorem 6.1] by contracting local bridges.

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
    retraction_depth: int = 1,
) -> ClassifierResult | None:
    """Classify by [FANP13, Theorem 6.2], or return ``None`` when it proves nothing.

    A successful certificate records the deleted edge.  Failure to classify a
    deleted-edge target falls through to the next inference rule.
    """
    for u, v in edges_in_no_triangle(graph):
        h = remove_edge_not_in_triangle(graph, u, v)
        h_behavior = classify_clique_behavior(
            h,
            tries=tries,
            bound=bound,
            extension_budget=extension_budget,
            retraction_depth=retraction_depth,
        )
        if h_behavior.verdict is Verdict.DIVERGENT:
            return (
                Verdict.DIVERGENT,
                (
                    "deleting an edge in no triangle yields a divergent graph "
                    "by [FANP13, Theorem 6.2]"
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
                    "divergent by [FANP13, Theorem 6.2]"
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

    The decision uses [FANP13, Theorem 6.2].

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
    retraction_depth: int = 1,
    max_candidates: int = 64,
) -> ClassifierResult | None:
    """Classify by an inverse cutpoint extension, or return ``None``.

    Searches for a local cutpoint whose admissible split (see
    :mod:`pycliques.cutpoints`) followed by removing the newly added edge
    ([FANP13, Theorem 6.2]) yields a graph with known clique behavior. The admissible
    split itself is justified by [FANP13, Theorem 6.1] applied to the constructed
    edge, not by any additional unverified hypothesis.

    ``extension_budget`` bounds how many times this (expensive) rule may
    fire along a single recursive classification chain; it is decremented
    only when this rule itself is applied, not by the cheaper [FANP13, Theorem 6.1] /
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
            retraction_depth=retraction_depth,
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
                    "an inverse cutpoint extension ([FANP13, Theorem 6.1]) followed by "
                    "deleting the added edge, contained in no triangle "
                    "([FANP13, Theorem 6.2]), yields a divergent graph"
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
                    "an inverse cutpoint extension ([FANP13, Theorem 6.1]) followed by "
                    "deleting the added edge ([FANP13, Theorem 6.2]) yields a "
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
    ([FANP13, Theorem 6.1]), such that deleting that edge -- contained in no triangle
    by construction requirements -- yields a graph of known clique behavior
    ([FANP13, Theorem 6.2]). See :mod:`pycliques.cutpoints` for the precise
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
    """Search for a [LNP06, Theorem 2.6] clockwork pair map into *target_graph*.

    Tries every ``m`` in ``2 .. max_m``; for each, collects up to
    ``max_coaffinations`` involutive ``(m + 1)``-coaffinations of
    ``target_graph``, then tries every ``n`` in ``0 .. max_n`` (subject to
    ``max_source_order``) against each candidate coaffination.

    ``m == 1`` (radius 2) is never tried: [LNP06, Theorem 3.1] only certifies that
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
    """Apply [LNP06, Theorems 2.6 and 3.1], returning None when inconclusive.

    Searches for an admissible graph morphism of coaffine pairs
    ``f: (R_{2m}^n, sigma) -> (graph, tau)`` for some ``m`` in
    ``2 .. max_m``, ``n`` in ``0 .. max_n``, and involutive
    ``(m + 1)``-coaffination ``tau`` of ``graph``.  By [LNP06, Theorem 3.1],
    ``R_{2m}^n`` is ``(m + 1)``-coaffine and rank divergent for ``m >= 2``;
    by [LNP06, Theorem 2.6], an admissible morphism to ``(graph, tau)`` transfers
    rank divergence to ``graph``, and rank divergence implies clique
    divergence.  ``m == 1`` (radius 2) is never searched: [LNP06, Theorem 3.1] does
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
                f"{{{n}}} ([LNP06, Theorem 3.1]) yielding rank divergence "
                "transferred by [LNP06, Theorem 2.6]"
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
                theorem_base_graph=graph,
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
            "([LNP06, Theorem 3.1]) into K(graph), yielding rank divergence of "
            "K(graph) by [LNP06, Theorem 2.6], hence rank divergence -- and clique "
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
            theorem_base_graph=graph,
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
    3.1 for :math:`m \\geq 2`.  An admissible morphism of coaffine pairs
    transfers rank divergence from the domain to the codomain by Theorem
    2.6, and rank divergence implies clique divergence.  ``m == 1`` (radius
    2) is never searched, since [LNP06, Theorem 3.1] does not certify rank
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
        bound, since [LNP06, Theorem 3.1] only certifies rank divergence of
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
        ``Verdict.CONVERGENT``.  Every positive result is re-checked by
        :func:`verify_clockwork_pair_map_certificate` (with ``clique_bound=bound``);
        a candidate that is ``INVALID`` or ``LIMIT_REACHED`` yields ``None``.
    """
    result = _classify_clockwork_pair_map(
        graph,
        max_m=max_m,
        max_n=max_n,
        max_coaffinations=max_coaffinations,
        max_source_order=max_source_order,
        bound=bound,
    )
    if result is None:
        return None
    certificate = result[2]
    if certificate is None:
        return None
    verification = verify_clockwork_pair_map_certificate(
        graph, certificate, clique_bound=bound
    )
    if verification.status is not CertificateVerificationStatus.VERIFIED:
        # An unverified or unverifiable candidate is never reported as proved.
        return None
    return result


def _valid_theorem_coaffination(
    graph: nx.Graph,
    coaffination: dict[Hashable, Hashable],
    radius: int,
) -> bool:
    """Check that a total involutive automorphism realizes the radius."""
    vertices = set(graph)
    if set(coaffination) != vertices:
        return False
    images = [coaffination[vertex] for vertex in graph]
    if not all(isinstance(image, Hashable) for image in images):
        return False
    if set(images) != vertices or len(set(images)) != len(vertices):
        return False
    if any(coaffination[coaffination[vertex]] != vertex for vertex in graph):
        return False
    edge_set = {frozenset(edge) for edge in graph.edges()}
    mapped_edges = {
        frozenset((coaffination[left], coaffination[right]))
        for left, right in graph.edges()
    }
    if mapped_edges != edge_set:
        return False
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    return all(
        distances[vertex].get(coaffination[vertex], math.inf) >= radius
        for vertex in graph
    )


def verify_clockwork_pair_map_certificate(
    original_graph: nx.Graph,
    certificate: Certificate,
    *,
    clique_bound: int | float | None = None,
    max_edges: int | None = None,
) -> CertificateVerificationResult:
    """Independently verify a [LNP06, Theorem 2.6] clockwork-pair certificate.

    ``theorem_base_graph`` records the exact graph searched by the pair-map
    rule. The standalone rule records ``original_graph``; the combined
    classifier may record its completely pared graph. A clique-graph
    certificate always targets the unpared ``K(theorem_base_graph)``.

    Clique-count or edge limits that prevent reconstructing that target
    produce ``LIMIT_REACHED`` rather than ``INVALID``.
    """
    if clique_bound is not None and clique_bound < 0:
        raise ValueError("clique_bound must be non-negative or None")
    if max_edges is not None and max_edges < 0:
        raise ValueError("max_edges must be non-negative or None")
    if (
        not isinstance(original_graph, nx.Graph)
        or original_graph.is_directed()
        or original_graph.is_multigraph()
    ):
        return CertificateVerificationResult(
            CertificateVerificationStatus.INVALID,
            "The original graph must be simple and undirected.",
        )

    def invalid(reason: str) -> CertificateVerificationResult:
        return CertificateVerificationResult(
            CertificateVerificationStatus.INVALID, reason
        )

    direct_rule = "theorem_2_6_clockwork_pair_map"
    clique_rule = "theorem_2_6_clockwork_pair_map_clique_graph"
    if certificate.rule not in {direct_rule, clique_rule}:
        return invalid(
            "Certificate rule is not a [LNP06, Theorem 2.6] clockwork-pair rule."
        )
    if (
        not isinstance(certificate.m, int)
        or not isinstance(certificate.n, int)
        or not isinstance(certificate.radius, int)
        or certificate.m < 2
        or certificate.n < 0
        or certificate.radius != certificate.m + 1
    ):
        return invalid("Certificate has invalid theorem parameters or radius.")
    if not isinstance(certificate.source, nx.Graph) or not isinstance(
        certificate.target, nx.Graph
    ):
        return invalid("Certificate is missing its source or target graph.")
    source = certificate.source
    target = certificate.target
    if (
        source.is_directed()
        or source.is_multigraph()
        or target.is_directed()
        or target.is_multigraph()
    ):
        return invalid("Theorem certificate graphs must be simple undirected graphs.")
    expected_source = clockwork_coaffine_pair(certificate.m, certificate.n)
    if not _same_labeled_graph(source, expected_source.graph):
        return invalid("Certificate source is not the canonical theorem clockwork.")
    sigma = certificate.source_coaffination
    if not isinstance(sigma, dict) or sigma != expected_source.coaffination:
        return invalid("Source coaffination is not the canonical theorem map.")
    if not _valid_theorem_coaffination(source, sigma, certificate.radius):
        return invalid("Source coaffination hypotheses are not satisfied.")

    tau = certificate.target_coaffination
    if not isinstance(tau, dict) or not _valid_theorem_coaffination(
        target, tau, certificate.radius
    ):
        return invalid("Target coaffination hypotheses are not satisfied.")
    mapping = certificate.map
    if not isinstance(mapping, dict):
        return invalid("Certificate map must be a total source-to-target mapping.")
    target_vertices = set(target)
    images = list(mapping.values())
    if not all(isinstance(image, Hashable) for image in images):
        return invalid("Certificate map contains a non-hashable image.")
    if set(mapping) != set(source) or not set(images) <= target_vertices:
        return invalid("Certificate map is not total or has images outside the target.")
    for left, right in source.edges():
        image_left, image_right = mapping[left], mapping[right]
        if image_left != image_right and not target.has_edge(image_left, image_right):
            return invalid("Certificate map does not preserve a source edge.")
    if any(mapping[sigma[vertex]] != tau[mapping[vertex]] for vertex in source):
        return invalid("Certificate map is not equivariant.")

    base = certificate.theorem_base_graph
    if not isinstance(base, nx.Graph) or base.is_directed() or base.is_multigraph():
        return invalid("Certificate is missing a valid theorem base graph.")
    pared = completely_pared_graph(original_graph)
    if not (
        _same_labeled_graph(base, original_graph) or _same_labeled_graph(base, pared)
    ):
        return invalid(
            "Theorem base graph is neither the original nor its pared graph."
        )

    if certificate.rule == direct_rule:
        if not _same_labeled_graph(target, base):
            return invalid(
                "Direct certificate target differs from its theorem base graph."
            )
    else:
        clique_target = clique_graph(
            base,
            math.inf if clique_bound is None else clique_bound,
            max_edges=max_edges,
        )
        if clique_target is None:
            return CertificateVerificationResult(
                CertificateVerificationStatus.LIMIT_REACHED,
                "Clique-graph reconstruction exceeded its configured bound.",
            )
        if not _same_labeled_graph(target, clique_target):
            return invalid(
                "Certificate target is not the unpared clique graph of its base."
            )

    return CertificateVerificationResult(
        CertificateVerificationStatus.VERIFIED,
        "[LNP06, Theorem 2.6] certificate and target provenance verified.",
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


def _test_clockwork(
    seq: CliqueSequence,
    tries: int,
    depth: int = 2,
    *,
    fallback_enabled: bool = False,
    fallback_max_order: int = 12,
    fallback_max_candidates: int = 5000,
) -> ClassifierResult | None:
    """Return a clockwork verdict, or ``None`` if this rule proves nothing."""
    for i in range(min(depth, tries)):
        g = seq[i]
        if g is None:
            return None
        recognized, structure_or_reason = recognize_clockwork(
            g,
            fallback_enabled=fallback_enabled,
            fallback_max_order=fallback_max_order,
            fallback_max_candidates=fallback_max_candidates,
        )
        if recognized:
            assert isinstance(structure_or_reason, ClockworkStructure)
            divergent, _ = is_clique_divergent_clockwork(
                g,
                structure=structure_or_reason,
                fallback_max_order=fallback_max_order,
                fallback_max_candidates=fallback_max_candidates,
                fallback_enabled=fallback_enabled,
            )
            if divergent is None:
                continue
            certificate = Certificate(
                rule="clockwork_at_iterate",
                target=g,
                target_status=(
                    "proven_divergent" if divergent else "proven_convergent"
                ),
                clockwork_structure=structure_or_reason,
                iterate_index=i,
            )
            if divergent is True:
                return (Verdict.DIVERGENT, "is clockwork divergent", certificate)
            return (Verdict.CONVERGENT, "is clockwork convergent", certificate)
    return None


def _special_octahedron_certificate(graph: nx.Graph, index: int) -> Certificate | None:
    """Attach an independently checkable witness to a positive search."""
    witness = find_special_octahedron_witness(graph)
    if witness is None:
        return None
    return Certificate.from_special_octahedron(graph, index, witness)


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
                _special_octahedron_certificate(g, i),
            )
    return None


def _make_clique_retraction_test(target: nx.Graph, label: str) -> Classifier:
    """Return a classifier that checks whether ``seq[1]`` retracts to *target*."""

    def _test(seq: CliqueSequence) -> ClassifierResult | None:
        g = seq[1]
        if g is not None and retracts(g, target):
            return (Verdict.DIVERGENT, label, None)
        return None

    return _test


def _test_iterate_certificates(
    seq: CliqueSequence,
    tries: int,
    retraction_depth: int,
    targets: tuple[tuple[nx.Graph, str], ...],
    *,
    octahedron_retract_iterates: tuple[int, ...] = (),
    octahedron_dimensions: tuple[int, ...] = (4,),
    octahedron_max_patterns: int = 5_000,
    octahedron_max_states: int = 100_000,
    octahedron_max_assignments: int = 500_000,
    octahedron_retract_presets: tuple[str, ...] = (),
) -> ClassifierResult | None:
    """Inspect each pared iterate fully before constructing the next.

    The previous separate Helly sweep could attempt K^(i+1)(G) before
    inspecting K^i(G) for a special octahedron. This exact per-iterate
    check avoids unnecessary clique-graph construction when a sufficient
    certificate is already present. Fixed-target retraction tests are
    restricted to the first two pared iterates, regardless of `tries`.
    An octahedral retraction search is opt-in at explicitly selected
    iterate indices and dimensions, with independent replay and budgets.
    A known target preset may construct its required iterate first.
    """
    octahedron_limited = False
    snub_seeded = "snub_o24" in octahedron_retract_presets and snub_o24_matches_source(
        seq[0]
    )
    # Try the supplied octahedral embedding before any general
    # Helly/special-octahedron recognition. No projection
    # is preloaded; the solver must discover a valid retraction.
    if snub_seeded:
        graph = seq[4]
        if graph is not None:
            prefix = [seq[j] for j in range(5)]
            if all(step is not None for step in prefix):
                pared_iterates = cast(list[nx.Graph], prefix)
                pairs = snub_o24_pairs_for_iterates(pared_iterates)
                if pairs is not None:
                    seeded = find_octahedral_retraction(
                        graph,
                        pairs,
                        max_states=octahedron_max_states,
                    )
                    if seeded.status is RetractionSearchStatus.FOUND:
                        if seeded.retraction is None or seeded.inclusion is None:
                            raise AssertionError("Missing seeded retraction witness")
                        target_vertices = [v for pair in pairs for v in pair]
                        target = graph.subgraph(target_vertices).copy()
                        seeded_witness = (seeded.retraction, seeded.inclusion)
                        if not verify_retraction_witness(graph, target, seeded_witness):
                            raise AssertionError("Seeded O24 witness failed replay")
                        seeded_certificate = Certificate.from_retraction(
                            graph,
                            target,
                            4,
                            seeded_witness,
                            rule="octahedron_retraction_seeded",
                            label="O_24",
                        )
                        return (
                            Verdict.DIVERGENT,
                            "pared iterate 4 retracts onto seeded O_24 (snub)",
                            seeded_certificate,
                        )
                    octahedron_limited |= (
                        seeded.status is RetractionSearchStatus.LIMIT_REACHED
                    )

    for i in range(tries):
        graph = seq[i]
        if graph is None:
            if octahedron_limited:
                return (
                    Verdict.INDETERMINATE,
                    "optional octahedral retraction search reached its resource limits",
                    None,
                )
            return None

        if is_clique_helly(graph):
            _logger.debug("Helly of index %s", i)
            return (Verdict.CONVERGENT, f"is eventually Helly (index {i})", None)

        if i < retraction_depth:
            for target, label in targets:
                witness = retracts(graph, target)
                if witness:
                    certificate = (
                        Certificate.from_retraction(graph, target, i, witness)
                        if isinstance(witness, tuple)
                        else None
                    )
                    return (
                        Verdict.DIVERGENT,
                        f"{label} (pared iterate {i})",
                        certificate,
                    )

        dim = special_octahedra_dimension(graph)
        if dim is not None:
            _logger.debug("Index %s has induced special octahedra", i)
            return (
                Verdict.DIVERGENT,
                f"eventually has a special octahedron (index {i}, dimension {dim})",
                _special_octahedron_certificate(graph, i),
            )

        if i in octahedron_retract_iterates:
            result = search_targeted_retractions(
                graph,
                families=("octahedra",),
                octahedron_dimensions=octahedron_dimensions,
                max_target_order=2 * max(octahedron_dimensions),
                max_patterns=octahedron_max_patterns,
                max_states=octahedron_max_states,
                max_assignments=octahedron_max_assignments,
            )["octahedra"]
            if result.status is TargetSearchStatus.FOUND:
                if not verify_targeted_retraction(graph, result):
                    raise AssertionError("Octahedral retraction witness failed replay")
                assert result.target_vertices is not None
                assert result.projection is not None
                assert result.target_kind is not None
                target = graph.subgraph(result.target_vertices).copy()
                dimension = int(result.target_kind.split(":")[1])
                certificate = Certificate.from_retraction(
                    graph,
                    target,
                    i,
                    (result.projection, {v: v for v in target}),
                    rule="octahedron_retraction",
                    label=f"O_{dimension}",
                )
                return (
                    Verdict.DIVERGENT,
                    f"pared iterate {i} retracts onto O_{dimension}",
                    certificate,
                )
            octahedron_limited |= result.status is TargetSearchStatus.LIMIT_REACHED
    if octahedron_limited:
        return (
            Verdict.INDETERMINATE,
            "optional octahedral retraction search reached its resource limits",
            None,
        )
    return None


@dataclass(frozen=True)
class _ClockworkLimits:
    fallback: bool = False
    max_order: int = 12
    max_candidates: int = 5000


@dataclass(frozen=True)
class _OctahedralSearch:
    iterates: tuple[int, ...] = ()
    dimensions: tuple[int, ...] = (4,)
    max_patterns: int = 5_000
    max_states: int = 100_000
    max_assignments: int = 500_000
    presets: tuple[str, ...] = ()


def _default_classifiers(
    tries: int,
    clockwork_depth: int = 2,
    *,
    retraction_depth: int = 1,
    clockwork: _ClockworkLimits = _ClockworkLimits(),
    octahedral: _OctahedralSearch = _OctahedralSearch(),
) -> list[Classifier]:
    """Check clockwork, then Helly/retractions/special octahedra per iterate.

    Every pared iterate is checked for all applicable certificates before
    attempting to generate the next one. Fixed-target retractions occur
    only on indices 0 and 1 (as selected by `retraction_depth`).
    """
    targets = (
        (suspension_of_cycle(5), "retracts to Susp(C_5)"),
        (suspension_of_cycle(6), "retracts to Susp(C_6)"),
        (suspension_of_cycle(7), "retracts to Susp(C_7)"),
        (complement_of_cycle(8), "retracts to Comp(C_8)"),
    )
    return [
        lambda seq: _test_clockwork(
            seq,
            tries,
            clockwork_depth,
            fallback_enabled=clockwork.fallback,
            fallback_max_order=clockwork.max_order,
            fallback_max_candidates=clockwork.max_candidates,
        ),
        lambda seq: _test_iterate_certificates(
            seq,
            tries,
            retraction_depth,
            targets,
            octahedron_retract_iterates=octahedral.iterates,
            octahedron_dimensions=octahedral.dimensions,
            octahedron_max_patterns=octahedral.max_patterns,
            octahedron_max_states=octahedral.max_states,
            octahedron_max_assignments=octahedral.max_assignments,
            octahedron_retract_presets=octahedral.presets,
        ),
    ]


def classify_clique_behavior(
    graph: nx.Graph,
    *,
    references: ReferenceSelection | None = None,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
    clockwork_depth: int = 2,
    retraction_depth: int = 1,
    clockwork_fallback: bool = False,
    clockwork_fallback_max_order: int = 12,
    clockwork_fallback_max_candidates: int = 5000,
    induced_clockwork_depth: int = 0,
    clockwork_retract_depth: int = 0,
    clockwork_max_order: int = 12,
    clockwork_max_candidates: int = 5000,
    clockwork_max_retraction_attempts: int = 20,
    clockwork_max_retraction_states: int = 100_000,
    clockwork_max_retraction_assignments: int = 1_000_000,
    octahedron_retract_iterates: tuple[int, ...] = (),
    octahedron_dimensions: tuple[int, ...] = (4,),
    octahedron_max_patterns: int = 5_000,
    octahedron_max_states: int = 100_000,
    octahedron_max_assignments: int = 500_000,
    octahedron_retract_presets: tuple[str, ...] = (),
) -> CliqueBehavior:
    """Classify the observed clique behavior of an undirected graph.

    This is the *fast pass*: it deliberately excludes the expensive
    [LNP06, Theorem 2.6] / coaffination search
    (:func:`classify_clockwork_pair_map`).  A graph left ``INDETERMINATE``
    by this fast pass may still be classifiable by that separate, more
    expensive rule; see :func:`classify_clique_behavior_with_clockwork_pair_search`
    to run both stages together.

    The input is completely pared before its iterated clique graphs are
    inspected.  The standard test suite certifies convergence through an
    eventually clique-Helly iterate, and divergence through clockwork,
    special-octahedron, and known-divergence criteria.

    The classifier applies a sequence of mathematically justified
    inference rules. These include retraction to registered reference
    graphs, [FANP13, Theorem 6.1] local-bridge contractions, [FANP13, Theorem 6.2]
    deletion of edges contained in no triangle, and theorem-motivated
    inverse cutpoint extensions. Each rule returns None when it
    establishes no verdict, allowing the pipeline to continue. A
    conjecturally divergent reference produces a conditional
    INDETERMINATE result rather than a proof of divergence.

    Each inference rule returns ``None`` when it establishes no verdict, so
    the pipeline continues.  The final ``INDETERMINATE`` result means that
    the available (fast) rules were exhausted without a definitive
    conclusion -- it is not evidence about what the (unrun) [LNP06, Theorem 2.6]
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
    references : tuple, optional
        Snapshot from select_reference_graphs, inherited by recursive calls.
    clockwork_depth : int, optional
        Number of iterates checked directly for clockwork structure
        (default: 2, preserving the historical G and K(G) check).
    retraction_depth : int, optional
        Number of pared iterates tested for retraction onto the four
        proven divergent targets. 1 checks only G; 2 checks G and K(G),
        both completely pared. Each iterate is checked first for Helly,
        then eligible retractions, then special octahedra; no subsequent
        iterate is constructed after finding a certificate.
    clockwork_fallback : bool, optional
        Enable bounded alternative-partition recognition in the direct
        clockwork check (default: False to preserve fast-pass cost).
    clockwork_fallback_max_order, clockwork_fallback_max_candidates
        Bounds for the optional direct-recognition fallback.
    induced_clockwork_depth : int, optional
        Number of initial iterates searched for induced divergent
        clockwork witnesses. This opt-in structural search does not by
        itself change the verdict (default: 0, disabled).
    clockwork_retract_depth : int, optional
        Number of initial iterates searched for verified retractions onto
        divergent induced clockwork subgraphs (default: 0, disabled).
    clockwork_max_order, clockwork_max_candidates
        Candidate bounds for the optional induced-clockwork searches.
    clockwork_max_retraction_attempts : int, optional
        Maximum divergent candidate targets tested per iterate.
    clockwork_max_retraction_states, clockwork_max_retraction_assignments
        Per-target limits on internal retraction backtracking.
    octahedron_retract_iterates : tuple[int, ...], optional
        Explicit indices of pared clique iterates to inspect for ordinary
        retractions onto O_d (default: empty, so no search at all).
        Use (3,) to inspect only K^3(G). Requires tries >= 4.
    octahedron_dimensions : tuple[int, ...], optional
        Exact dimensions d >= 3 to test (default: (4,)).
    octahedron_max_patterns, octahedron_max_states, octahedron_max_assignments
        Positive bounds for each selected iterate's candidate and CSP search.
        Reaching a limit never establishes the absence of a retraction.
    octahedron_retract_presets : tuple[str, ...], optional
        Opt-in known-embedding search. Currently "snub_o24" is supported:
        for a snub disphenoid (any vertex labeling), construct K^4(G),
        transport the independently known induced O_24 embedding, and
        search afresh for a retraction using the specialized bitset solver.
        Requires tries >= 5 and bound >= 1076. This does not replay a saved
        projection and does not discover the O_24 embedding from scratch.
        The per-target budget is octahedron_max_states; the generic pattern
        and assignment budgets do not apply to this seeded solver.

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
    selection = (
        current_references(_REFERENCE_GRAPHS.values())
        if references is None
        else references
    )
    with reference_scope(selection):
        if tries < 1:
            raise ValueError("tries must be at least 1")
        if bound < 1:
            raise ValueError("bound must be at least 1")
        if clockwork_depth < 0:
            raise ValueError("clockwork_depth must be non-negative")
        if retraction_depth not in (0, 1, 2):
            raise ValueError("retraction_depth must be 0, 1, or 2")
        if clockwork_fallback_max_order < 0:
            raise ValueError("clockwork_fallback_max_order must be non-negative")
        if clockwork_fallback_max_candidates < 1:
            raise ValueError("clockwork_fallback_max_candidates must be at least 1")
        if induced_clockwork_depth < 0:
            raise ValueError("induced_clockwork_depth must be non-negative")
        if clockwork_retract_depth < 0:
            raise ValueError("clockwork_retract_depth must be non-negative")
        if clockwork_max_order < 0:
            raise ValueError("clockwork_max_order must be non-negative")
        if clockwork_max_candidates < 1:
            raise ValueError("clockwork_max_candidates must be at least 1")
        if clockwork_max_retraction_attempts < 1:
            raise ValueError("clockwork_max_retraction_attempts must be at least 1")
        if clockwork_max_retraction_states < 1:
            raise ValueError("clockwork_max_retraction_states must be at least 1")
        if clockwork_max_retraction_assignments < 1:
            raise ValueError("clockwork_max_retraction_assignments must be at least 1")
        if any(
            type(i) is not int or i < 0 or i >= tries
            for i in octahedron_retract_iterates
        ) or len(set(octahedron_retract_iterates)) != len(octahedron_retract_iterates):
            raise ValueError(
                "octahedron_retract_iterates must be distinct indices < tries"
            )
        if (
            not octahedron_dimensions
            or any(type(d) is not int or d < 3 for d in octahedron_dimensions)
            or len(set(octahedron_dimensions)) != len(octahedron_dimensions)
        ):
            raise ValueError("octahedron_dimensions must be distinct integers >= 3")
        if (
            min(
                octahedron_max_patterns,
                octahedron_max_states,
                octahedron_max_assignments,
            )
            < 1
        ):
            raise ValueError("octahedron search budgets must be positive")
        if len(set(octahedron_retract_presets)) != len(
            octahedron_retract_presets
        ) or any(preset != "snub_o24" for preset in octahedron_retract_presets):
            raise ValueError("Unknown or duplicated octahedron retraction preset")
        if "snub_o24" in octahedron_retract_presets and (tries < 5 or bound < 1076):
            raise ValueError("snub_o24 requires tries >= 5 and bound >= 1076")

        pared_graph = completely_pared_graph(graph)
        seq = CliqueSequence(pared_graph, bound=bound)
        conditional_result: ClassifierResult | None = None

        # Opt-in research replay must execute the solver rather than stop at
        # the independently proven snub reference classification.
        seeded_snub_requested = (
            "snub_o24" in octahedron_retract_presets
            and snub_o24_matches_source(pared_graph)
        )
        reference_result = (
            None if seeded_snub_requested else _classify_reference_graph(pared_graph)
        )
        if reference_result is not None:
            verdict, reason, certificate = reference_result
            if verdict is Verdict.INDETERMINATE:
                if conditional_result is None:
                    conditional_result = reference_result
            else:
                return CliqueBehavior(
                    verdict, reason, seq.graph_count, False, pared_graph, certificate
                )

        theorem_4_6_result = _classify_suspension_2_coaffination(graph)
        if theorem_4_6_result is not None:
            verdict, reason, certificate = theorem_4_6_result
            if verdict is Verdict.INDETERMINATE:
                if conditional_result is None:
                    conditional_result = theorem_4_6_result
            else:
                return CliqueBehavior(
                    verdict, reason, seq.graph_count, False, pared_graph, certificate
                )

        for classifier in _default_classifiers(
            tries,
            clockwork_depth,
            retraction_depth=retraction_depth,
            clockwork=_ClockworkLimits(
                clockwork_fallback,
                clockwork_fallback_max_order,
                clockwork_fallback_max_candidates,
            ),
            octahedral=_OctahedralSearch(
                octahedron_retract_iterates,
                octahedron_dimensions,
                octahedron_max_patterns,
                octahedron_max_states,
                octahedron_max_assignments,
                octahedron_retract_presets,
            ),
        ):
            result = classifier(seq)
            if result is not None:
                verdict, reason, certificate = result
                if verdict is Verdict.INDETERMINATE:
                    if conditional_result is None:
                        conditional_result = result
                else:
                    return CliqueBehavior(
                        verdict,
                        reason,
                        seq.graph_count,
                        False,
                        pared_graph,
                        certificate,
                    )

        reference_dependency = _classify_reference_dependency(pared_graph)
        if reference_dependency is not None:
            verdict, reason, certificate = reference_dependency
            if verdict is Verdict.INDETERMINATE:
                if conditional_result is None:
                    conditional_result = reference_dependency
            else:
                return CliqueBehavior(
                    verdict, reason, seq.graph_count, False, pared_graph, certificate
                )

        local_bridge_result = _classify_local_bridge(
            pared_graph,
            tries=tries,
            bound=bound,
            extension_budget=extension_budget,
            retraction_depth=retraction_depth,
        )
        if local_bridge_result is not None:
            verdict, reason, certificate = local_bridge_result
            if verdict is Verdict.INDETERMINATE:
                if conditional_result is None:
                    conditional_result = local_bridge_result
            else:
                return CliqueBehavior(
                    verdict, reason, seq.graph_count, False, pared_graph, certificate
                )

        non_triangle_edge_result = _classify_non_triangle_edge(
            pared_graph,
            tries=tries,
            bound=bound,
            extension_budget=extension_budget,
            retraction_depth=retraction_depth,
        )
        if non_triangle_edge_result is not None:
            verdict, reason, certificate = non_triangle_edge_result
            if verdict is Verdict.INDETERMINATE:
                if conditional_result is None:
                    conditional_result = non_triangle_edge_result
            else:
                return CliqueBehavior(
                    verdict, reason, seq.graph_count, False, pared_graph, certificate
                )

        inverse_extension_result = _classify_inverse_cutpoint_extension(
            pared_graph,
            tries=tries,
            bound=bound,
            extension_budget=extension_budget,
            retraction_depth=retraction_depth,
        )
        if inverse_extension_result is not None:
            verdict, reason, certificate = inverse_extension_result
            if verdict is Verdict.INDETERMINATE:
                if conditional_result is None:
                    conditional_result = inverse_extension_result
            else:
                return CliqueBehavior(
                    verdict, reason, seq.graph_count, False, pared_graph, certificate
                )

        observations: list[InducedClockworkWitness] = []
        search_limited = False
        for i in range(clockwork_retract_depth):
            iterate = seq[i]
            if iterate is None:
                break
            iterate_prefix: list[nx.Graph] = []
            for iterate_number in range(i + 1):
                cached = seq[iterate_number]
                if cached is None:
                    break
                iterate_prefix.append(cached)
            if len(iterate_prefix) != i + 1:
                break
            retraction_search = find_divergent_clockwork_retract(
                iterate,
                max_target_order=clockwork_max_order,
                max_candidates=clockwork_max_candidates,
                max_retraction_attempts=clockwork_max_retraction_attempts,
                max_retraction_states=clockwork_max_retraction_states,
                max_retraction_assignments=clockwork_max_retraction_assignments,
                recognition_max_order=clockwork_fallback_max_order,
                recognition_max_candidates=clockwork_fallback_max_candidates,
                replay_clique_bound=bound,
                iterate_index=i,
                origin_graph=pared_graph,
                iterate_graphs=tuple(iterate_prefix),
            )
            if retraction_search.certificate is not None:
                certificate = Certificate(
                    rule="induced_clockwork_retraction",
                    target=retraction_search.certificate.target_graph,
                    target_status="proven_divergent",
                    map=retraction_search.certificate.retraction_map,
                    source=iterate,
                    clockwork_structure=retraction_search.certificate.structure,
                    iterate_index=i,
                )
                return CliqueBehavior(
                    Verdict.DIVERGENT,
                    f"iterate {i} retracts onto a divergent clockwork graph",
                    seq.graph_count,
                    seq.exhausted,
                    pared_graph,
                    certificate,
                    tuple(observations),
                )
            search_limited |= (
                retraction_search.status is ClockworkSearchStatus.LIMIT_REACHED
            )

        for i in range(induced_clockwork_depth):
            iterate = seq[i]
            if iterate is None:
                break
            induced_search = find_induced_clockwork_subgraphs(
                iterate,
                max_order=clockwork_max_order,
                max_candidates=clockwork_max_candidates,
                divergent_only=True,
                iterate_index=i,
                recognition_max_order=clockwork_fallback_max_order,
                recognition_max_candidates=clockwork_fallback_max_candidates,
            )
            observations.extend(induced_search.witnesses)
            search_limited |= (
                induced_search.status is ClockworkSearchStatus.LIMIT_REACHED
            )

        bound_exceeded = seq.exhausted
        if observations:
            indices = sorted({witness.iterate_index for witness in observations})
            reason = (
                "induced divergent clockwork found at iterate(s) "
                f"{indices}; induced containment alone does not certify divergence"
            )
        elif search_limited:
            reason = (
                "clockwork search resource limit reached; "
                "behavior remains indeterminate"
            )
        else:
            reason = (
                "clique count exceeded bound"
                if bound_exceeded
                else "behavior is indeterminate under the available tests"
            )
        if conditional_result is not None:
            _, conditional_reason, conditional_certificate = conditional_result
            return CliqueBehavior(
                Verdict.INDETERMINATE,
                conditional_reason,
                seq.graph_count,
                bound_exceeded,
                pared_graph,
                conditional_certificate,
                observations=tuple(observations),
            )
        return CliqueBehavior(
            Verdict.INDETERMINATE,
            reason,
            seq.graph_count,
            bound_exceeded,
            pared_graph,
            observations=tuple(observations),
        )


def classify_clique_behavior_with_clockwork_pair_search(
    graph: nx.Graph,
    *,
    references: ReferenceSelection | None = None,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
    extension_budget: int = 1,
    clockwork_depth: int = 2,
    clockwork_fallback: bool = False,
    clockwork_fallback_max_order: int = 12,
    clockwork_fallback_max_candidates: int = 5000,
    induced_clockwork_depth: int = 0,
    clockwork_retract_depth: int = 0,
    clockwork_max_order: int = 12,
    clockwork_max_candidates: int = 5000,
    clockwork_max_retraction_attempts: int = 20,
    clockwork_max_retraction_states: int = 100_000,
    clockwork_max_retraction_assignments: int = 1_000_000,
    octahedron_retract_iterates: tuple[int, ...] = (),
    octahedron_dimensions: tuple[int, ...] = (4,),
    octahedron_max_patterns: int = 5_000,
    octahedron_max_states: int = 100_000,
    octahedron_max_assignments: int = 500_000,
    octahedron_retract_presets: tuple[str, ...] = (),
    max_m: int = 3,
    max_n: int = 3,
    max_coaffinations: int = 20,
    max_source_order: int = 40,
    clockwork_pair_bound: int | None = None,
) -> CliqueBehavior:
    """Run the fast pass, then [LNP06, Theorem 2.6] on unresolved graphs.

    This is the two-stage workflow used by the ``small-behavior-clockwork-pairs``
    CLI: :func:`classify_clique_behavior` (the fast pass) is run first, and
    :func:`classify_clockwork_pair_map` (the [LNP06, Theorem 2.6] / coaffination
    search) is only attempted when the fast pass leaves the graph
    ``INDETERMINATE``.  Graphs already classified as ``CONVERGENT`` or
    ``DIVERGENT`` never reach the expensive search.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries, bound, extension_budget and the ``clockwork_*`` search options
        Forwarded to :func:`classify_clique_behavior`.
    max_m, max_n, max_coaffinations, max_source_order
        Forwarded to :func:`classify_clockwork_pair_map`.
    clockwork_pair_bound : int, optional
        Clique-graph bound used by the [LNP06, Theorem 2.6] search's ``K(graph)``
        fallback.  Defaults to ``bound`` when not given.

    .. rubric:: Returns

    CliqueBehavior
        The fast-pass result unchanged, unless the fast pass was
        ``INDETERMINATE`` and [LNP06, Theorem 2.6] establishes ``DIVERGENT``.
    """
    fast = classify_clique_behavior(
        graph,
        references=references,
        tries=tries,
        bound=bound,
        extension_budget=extension_budget,
        clockwork_depth=clockwork_depth,
        clockwork_fallback=clockwork_fallback,
        clockwork_fallback_max_order=clockwork_fallback_max_order,
        clockwork_fallback_max_candidates=clockwork_fallback_max_candidates,
        induced_clockwork_depth=induced_clockwork_depth,
        clockwork_retract_depth=clockwork_retract_depth,
        clockwork_max_order=clockwork_max_order,
        clockwork_max_candidates=clockwork_max_candidates,
        clockwork_max_retraction_attempts=clockwork_max_retraction_attempts,
        clockwork_max_retraction_states=clockwork_max_retraction_states,
        clockwork_max_retraction_assignments=clockwork_max_retraction_assignments,
        octahedron_retract_iterates=octahedron_retract_iterates,
        octahedron_dimensions=octahedron_dimensions,
        octahedron_max_patterns=octahedron_max_patterns,
        octahedron_max_states=octahedron_max_states,
        octahedron_max_assignments=octahedron_max_assignments,
        octahedron_retract_presets=octahedron_retract_presets,
    )
    if fast.verdict is not Verdict.INDETERMINATE:
        return fast

    theorem_bound = bound if clockwork_pair_bound is None else clockwork_pair_bound
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
        fast.observations,
    )


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


if __name__ == "__main__":  # pragma: no cover
    main()
