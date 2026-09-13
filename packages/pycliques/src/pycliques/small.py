"""Determine the clique behavior of small graphs."""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

import networkx as nx
from rich.logging import RichHandler

from pycliques import __version__
from pycliques.cliques import clique_graph
from pycliques.clockwork import is_clique_divergent_clockwork, recognize_clockwork
from pycliques.cutpoints import (
    contract_local_bridge,
    edges_in_no_triangle,
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
    """Structured provenance for a classification decision."""

    rule: str
    target: nx.Graph | None = None
    target_status: str | None = None
    target_label: str | None = None
    map: tuple[dict, dict] | dict | None = None


@dataclass(frozen=True)
class CliqueBehavior:
    """Result of classifying a graph under clique-graph iteration.

    ``INDETERMINATE`` means that the available sufficient tests did not
    decide the behavior within the configured limits.  It is not a
    mathematical assertion that the graph is neither convergent nor
    divergent.

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
        Structured justification for the verdict when the classifier uses a
        retraction or reduction-based inference rule.
    """

    verdict: Verdict
    reason: str
    iterations_checked: int
    bound_exceeded: bool
    pared_graph: nx.Graph
    certificate: Certificate | None = None


#: Type alias for a classifier function.
ClassifierResult = tuple[Verdict, str] | None
Classifier = Callable[["CliqueSequence"], ClassifierResult]


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

    def __init__(self, graph: nx.Graph, bound: int = 30) -> None:
        self._graphs: list[nx.Graph] = [graph]
        self._bound = bound
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
) -> tuple[Verdict, str, Certificate | None] | None:
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
) -> tuple[Verdict, str, Certificate | None] | None:
    """Return a retraction certificate when a known reference graph is involved."""
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
) -> tuple[Verdict, str, Certificate | None] | None:
    """Classify clique behavior by contracting local bridges.

    The decision uses Theorem 6.1.
    """
    for u, v in local_bridges(graph):
        h = contract_local_bridge(graph, u, v)
        h_behavior = classify_clique_behavior(h, tries=tries, bound=bound)
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
                ),
            )
    return None


def classify_local_bridge(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
) -> tuple[Verdict, str, Certificate | None] | None:
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
        A tuple of ``(verdict, reason, certificate)`` if a local bridge exists and
        the clique behavior of the contracted graph can be classified; ``None``
        otherwise.
    """
    return _classify_local_bridge(graph, tries=tries, bound=bound)


def _classify_non_triangle_edge(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
) -> tuple[Verdict, str, Certificate | None] | None:
    """Classify clique behavior by deleting edges in no triangle.

    The decision uses Theorem 6.2.
    """
    for u, v in edges_in_no_triangle(graph):
        h = remove_edge_not_in_triangle(graph, u, v)
        h_behavior = classify_clique_behavior(h, tries=tries, bound=bound)
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
                ),
            )
    return None


def classify_non_triangle_edge(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
) -> tuple[Verdict, str, Certificate | None] | None:
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
        otherwise.
    """
    return _classify_non_triangle_edge(graph, tries=tries, bound=bound)


def _test_eventually_helly(seq: CliqueSequence, tries: int) -> ClassifierResult:
    """Convergent if some iterate is clique-Helly."""
    for i in range(tries):
        g = seq[i]
        if g is None:
            return None
        if is_clique_helly(g):
            _logger.debug(f"Helly of index {i}")
            return (Verdict.CONVERGENT, f"is eventually Helly (index {i})")
    return None


def _test_clockwork(seq: CliqueSequence, tries: int) -> ClassifierResult:
    """Check clockwork recognition on seq[0] and seq[1]."""
    for i in range(min(2, tries)):
        g = seq[i]
        if g is None:
            return None
        if recognize_clockwork(g)[0]:
            divergent, _ = is_clique_divergent_clockwork(g)
            if divergent is True:
                return (Verdict.DIVERGENT, "is clockwork divergent")
            if divergent is False:
                return (Verdict.CONVERGENT, "is clockwork convergent")
    return None


def _test_eventually_special_octahedra(
    seq: CliqueSequence, tries: int
) -> ClassifierResult:
    """Divergent if some iterate contains a special octahedron."""
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
            )
    return None


def _make_retraction_test(target: nx.Graph, label: str) -> Classifier:
    """Return a classifier that checks whether ``seq[0]`` retracts to *target*."""

    def _test(seq: CliqueSequence) -> ClassifierResult:
        g = seq[0]
        if g is not None and retracts(g, target):
            return (Verdict.DIVERGENT, label)
        return None

    return _test


def _default_classifiers(tries: int) -> list[Classifier]:
    """Return the standard ordered suite of clique-behavior tests."""
    return [
        lambda seq: _test_clockwork(seq, tries),
        lambda seq: _test_eventually_helly(seq, tries),
        lambda seq: _test_eventually_special_octahedra(seq, tries),
        _make_retraction_test(suspension_of_cycle(5), "retracts to Susp(C_5)"),
        _make_retraction_test(suspension_of_cycle(6), "retracts to Susp(C_6)"),
        _make_retraction_test(suspension_of_cycle(7), "retracts to Susp(C_7)"),
        _make_retraction_test(complement_of_cycle(8), "retracts to Comp(C_8)"),
    ]


def classify_clique_behavior(
    graph: nx.Graph,
    *,
    tries: int = _MAX_ITERATIONS,
    bound: int = 30,
) -> CliqueBehavior:
    """Classify the observed clique behavior of an undirected graph.

    The input is completely pared before its iterated clique graphs are
    inspected.  The standard test suite certifies convergence through an
    eventually clique-Helly iterate, and divergence through clockwork,
    special-octahedron, and known-divergence criteria.

    Two distinct inference rules are recognized when a graph can be
    connected to a known reference graph through a mathematically justified
    relation:

        * ordinary retraction: ``retracts(G, H)`` and ``H`` is clique divergent.

    A conjecturally divergent reference graph yields a conditional
    ``INDETERMINATE`` result rather than a proof of divergence.

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
    seq = CliqueSequence(pared_graph, bound=bound)

    reference_result = _classify_reference_graph(pared_graph)
    if reference_result is not None:
        verdict, reason, certificate = reference_result
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    for classifier in _default_classifiers(tries):
        result = classifier(seq)
        if result is not None:
            verdict, reason = result
            return CliqueBehavior(verdict, reason, seq.graph_count, False, pared_graph)

    reference_dependency = _classify_reference_dependency(pared_graph)
    if reference_dependency is not None:
        verdict, reason, certificate = reference_dependency
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    local_bridge_result = _classify_local_bridge(pared_graph, tries=tries, bound=bound)
    if local_bridge_result is not None:
        verdict, reason, certificate = local_bridge_result
        return CliqueBehavior(
            verdict, reason, seq.graph_count, False, pared_graph, certificate
        )

    non_triangle_edge_result = _classify_non_triangle_edge(
        pared_graph, tries=tries, bound=bound
    )
    if non_triangle_edge_result is not None:
        verdict, reason, certificate = non_triangle_edge_result
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


def _make_clique_retraction_test(target: nx.Graph, label: str) -> Classifier:
    """Return a classifier that checks whether ``seq[1]`` retracts to *target*."""

    def _test(seq: CliqueSequence) -> ClassifierResult:
        g = seq[1]
        if g is not None and retracts(g, target):
            return (Verdict.DIVERGENT, label)
        return None

    return _test


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
) -> None:
    """Save indeterminate pared graphs to a human-readable file.

    Each line contains the original graph index, the order of the pared
    graph, its graph6 string, and certificate metadata when available.

    If the file already exists, the new results are merged with the
    existing entries.  Duplicate indices are resolved in favour of the
    new run so that re-processing a range always updates the record.
    """
    path = _indeterminate_file_path(order, data_dir)

    # Load existing entries keyed by original index.
    existing: dict[int, str] = {}
    if path.is_file():
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


def is_eventually_helly(graph: nx.Graph, tries: int = 8, bound: int = 30) -> bool:
    """Return whether ``graph`` is eventually clique-Helly.

    Starting from ``graph``, repeatedly compute the completely-pared clique
    graph. Return ``True`` as soon as one iterate is clique-Helly.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    tries : int
        Maximum number of clique-graph iterations (default 8).
    bound : int
        Maximum number of cliques allowed before aborting (default 30).

    .. rubric:: Returns

    bool
        ``True`` if an iterated clique graph within ``tries`` steps is
        clique-Helly, ``False`` otherwise.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.helly import is_clique_helly
    >>> from pycliques.small import is_eventually_helly
    >>> is_clique_helly(nx.triangular_lattice_graph(3,3))
    False
    >>> is_eventually_helly(nx.triangular_lattice_graph(3,3))
    True

    """
    seq = CliqueSequence(graph, bound=bound)
    return _test_eventually_helly(seq, tries + 1) is not None


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


if __name__ == "__main__":  # pragma: no cover
    main()
