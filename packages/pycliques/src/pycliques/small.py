"""Determine the clique behavior of small graphs."""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path

import networkx as nx
from rich.logging import RichHandler

from pycliques import __version__
from pycliques.cliques import clique_graph
from pycliques.clockwork import is_clique_divergent_clockwork, recognize_clockwork
from pycliques.dominated import completely_pared_graph, find_dominated_vertex
from pycliques.helly import is_clique_helly
from pycliques.named import complement_of_cycle, suspension_of_cycle
from pycliques.retractions import retracts, special_octahedra_dimension

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict enum and CliqueSequence
# ---------------------------------------------------------------------------


class Verdict(Enum):
    """Classification outcome for a graph's clique behavior."""

    CONVERGENT = auto()
    DIVERGENT = auto()


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


def _test_eventually_helly(seq: CliqueSequence) -> ClassifierResult:
    """Convergent if some iterate is clique-Helly."""
    for i in range(_MAX_ITERATIONS):
        g = seq[i]
        if g is None:
            return None
        if is_clique_helly(g):
            _logger.debug(f"Helly of index {i}")
            return (Verdict.CONVERGENT, f"is eventually Helly (index {i})")
    return None


def _test_clockwork(seq: CliqueSequence) -> ClassifierResult:
    """Check clockwork recognition on seq[0] and seq[1]."""
    for i in range(2):
        g = seq[i]
        if g is None:
            return None
        if recognize_clockwork(g)[0]:
            if is_clique_divergent_clockwork(g):
                return (Verdict.DIVERGENT, "is clockwork divergent")
            return (Verdict.CONVERGENT, "is clockwork convergent")
    return None


def _test_eventually_special_octahedra(seq: CliqueSequence) -> ClassifierResult:
    """Divergent if some iterate contains a special octahedron."""
    for i in range(_MAX_ITERATIONS):
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


def _save_indeterminate(
    order: int,
    indeterminate: list[tuple[int, nx.Graph]],
    data_dir: Path,
) -> None:
    """Save indeterminate pared graphs to a human-readable file.

    Each line contains the original graph index, the order of the pared
    graph, and its graph6 string.

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
                parts = stripped.split(maxsplit=2)
                existing[int(parts[0])] = stripped
        _logger.info(
            f"Found existing file with {len(existing)} entries; merging new results"
        )

    # Build lines for the new entries (overwrite any duplicate index).
    for idx, graph in indeterminate:
        g = nx.convert_node_labels_to_integers(graph)
        g6 = nx.to_graph6_bytes(g, header=False).decode("ascii").strip()
        existing[idx] = f"{idx} {g.order()} {g6}"

    # Write everything back sorted by index.
    with path.open("w", encoding="utf-8") as f:
        f.write(
            f"# Indeterminate clique behavior - connected graphs of order {order}\n"
        )
        f.write("# Format: original_index pared_order graph6\n")
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
                parts = line.split()
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
    i = 0
    while not is_clique_helly(graph) and i < tries:
        i = i + 1
        graph = clique_graph(graph, bound)
        if graph is None:
            return False
        else:
            graph = completely_pared_graph(graph)
    if is_clique_helly(graph):
        _logger.debug(f"Helly of index {i}")
        return True
    else:
        return False


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
    g_curr = graph
    for i in range(tries):
        if special_octahedra_dimension(g_curr) is not None:
            _logger.debug(f"Index {i} has induced special octahedra")
            return True
        g_curr = clique_graph(g_curr, bound)
        if g_curr is None:
            return None
        g_curr = completely_pared_graph(g_curr)
    return None


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

    # 1. Build the classifier pipeline
    _logger.info("Precomputing target mathematical structures...")
    classifiers: list[Classifier] = [
        _test_eventually_helly,
        _test_clockwork,
        _test_eventually_special_octahedra,
        _make_retraction_test(suspension_of_cycle(5), "retracts to Susp(C_5)"),
        _make_retraction_test(suspension_of_cycle(6), "retracts to Susp(C_6)"),
        _make_retraction_test(complement_of_cycle(8), "retracts to Comp(C_8)"),
        # _make_clique_retraction_test(
        #     complement_of_cycle(10), "clique graph retracts to Comp(C_10)"
        # ),
    ]

    # Load previously saved indeterminate graphs for smaller orders
    known_indeterminate: dict[int, list[nx.Graph]] = {}
    if lookup:
        known_indeterminate = _load_indeterminate_graphs(order, data_dir)

    convergent: list[int] = []
    divergent: list[int] = []
    further: list[int] = []
    further_pared: list[tuple[int, nx.Graph]] = []
    further_graphs: list[tuple[int, nx.Graph]] = []
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

                    graph = completely_pared_graph(graph)

                    behavior = _classify_graph(
                        graph,
                        classifiers,
                        known_indeterminate,
                        index,
                        convergent,
                        divergent,
                        further,
                        further_pared,
                        further_graphs,
                        verdict_file,
                    )
                    _logger.debug(f"Graph {index}: {behavior}")

    total_processed = len(convergent) + len(divergent) + len(further) + len(reducible)
    _logger.info(f"Analysis Complete! Processed {total_processed} total graphs.")
    _logger.info(f"Indices that deserve further study: {further}")
    _logger.info(f"Total convergent graphs: {len(convergent)}")
    _logger.info(f"Total divergent graphs: {len(divergent)}")
    _logger.info(f"Total reducible graphs skipped: {len(reducible)}")
    _logger.info(f"Total graphs reduced to unknown: {len(further_pared)}")
    _logger.info(f"Total unknown graphs (further study): {len(further)}")

    if save and further_graphs:
        _save_indeterminate(order, further_graphs, data_dir)


def _classify_graph(
    graph: nx.Graph,
    classifiers: list[Classifier],
    known_indeterminate: dict[int, list[nx.Graph]],
    index: int,
    convergent: list[int],
    divergent: list[int],
    further: list[int],
    further_pared: list[tuple[int, nx.Graph]],
    further_graphs: list[tuple[int, nx.Graph]],
    verdict_file=None,
) -> str:
    """Run the classifier pipeline on a single pared graph.

    Returns the behavior description string for logging.
    If *verdict_file* is not ``None``, writes a tab-separated line
    ``index\\tverdict\\treason`` for every graph processed.
    """

    def _write_verdict(verdict_label: str, reason: str) -> None:
        if verdict_file is not None:
            verdict_file.write(f"{index}\t{verdict_label}\t{reason}\n")

    if known_indeterminate and _is_known_indeterminate(graph, known_indeterminate):
        further_pared.append((index, graph))
        _write_verdict("UNKNOWN", "reduces to known indeterminate graph")
        return "reduces to known indeterminate graph"

    seq = CliqueSequence(graph)
    for classifier in classifiers:
        result = classifier(seq)
        if result is not None:
            verdict, behavior = result
            if verdict is Verdict.CONVERGENT:
                convergent.append(index)
                _write_verdict("CONVERGENT", behavior)
            else:
                divergent.append(index)
                _write_verdict("DIVERGENT", behavior)
            return behavior

    further.append(index)
    further_graphs.append((index, graph))
    _write_verdict("UNKNOWN", "has character unknown so far")
    return "has character unknown so far"


def main():  # pragma: no cover
    """Entry point for the ``clique-behavior`` console script."""
    _main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
