"""Command-line entry points for the graph census and the clockwork second pass."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import networkx as nx
from rich.logging import RichHandler

from . import __version__
from . import small as rules
from .classification_types import Certificate, Verdict
from .clique_sequence import CliqueSequence
from .dominated import find_dominated_vertex
from .indeterminate_files import (
    _graph6_string,
    _indeterminate_file_path,
    _is_known_indeterminate,
    _load_indeterminate_file_entries_with_metadata,
    _load_indeterminate_graphs,
    _save_indeterminate,
    _update_processed_indeterminate,
)
from .named import complement_of_cycle

_logger = logging.getLogger("pycliques.small")
_DEFAULT_DATA_DIR = Path(".")


def _connected_records(order: int) -> Iterator[tuple[int, str]]:
    try:
        from pyg6data import iter_graph6_records
    except ModuleNotFoundError as error:
        if error.name != "pyg6data":
            raise
        raise SystemExit(
            "Install pycliques[data] to read the bundled graph census"
        ) from error
    try:
        return iter_graph6_records(order)
    except ValueError as error:
        _logger.error(f"Error: Internal data for order {order} not available.")
        raise SystemExit(1) from error


def _recheck_indeterminate_file(
    entries: list[tuple[int, nx.Graph, Certificate | None]],
    bound: int,
    classifier: rules.Classifier | None = None,
) -> list[tuple[int, nx.Graph, Certificate | None]]:
    """Re-run a classifier on graphs saved as indeterminate."""
    still_indeterminate: list[tuple[int, nx.Graph, Certificate | None]] = []
    for index, graph, saved_certificate in entries:
        _logger.info("Studying graph %s (%s)", index, _graph6_string(graph))
        if classifier is None:
            behavior = rules.classify_clique_behavior(graph, bound=bound)
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
            recheck_classifier = rules._make_clique_retraction_test(
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
    processed_indices: set[int] = set()

    range_msg = ""
    if start is not None or end is not None:
        lo = start if start is not None else 0
        hi = end if end is not None else "last"
        range_msg = f" (indices {lo}-{hi})"
    _logger.info(
        f"Beginning analysis of connected graphs of order {order}{range_msg}..."
    )

    records = _connected_records(order)

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

        for index, line in records:
            if start is not None and index < start:
                continue
            if end is not None and index > end:
                break
            processed_indices.add(index)

            if index > 0 and index % 10000 == 0:  # pragma: no cover
                _logger.info(f"Processed up to index {index}...")

            assert isinstance(line, str)
            graph = nx.from_graph6_bytes(line.encode("ascii"))

            if skip_dominated and find_dominated_vertex(graph) is not None:
                reducible.append(index)
                if verdict_file is not None:
                    verdict_file.write(f"{index}\tREDUCIBLE\thas dominated vertices\n")
                _logger.debug(f"Graph {index}: has dominated vertices")
                continue

            result = rules.classify_clique_behavior(graph, bound=bound)
            if result.verdict is Verdict.CONVERGENT:
                convergent.append(index)
                verdict_label = "CONVERGENT"
                reason = result.reason
            elif result.verdict is Verdict.DIVERGENT:
                divergent.append(index)
                verdict_label = "DIVERGENT"
                reason = result.reason
            else:
                further.append(index)
                further_graphs.append((index, result.pared_graph, result.certificate))
                verdict_label = "INDETERMINATE"
                if lookup and _is_known_indeterminate(
                    result.pared_graph, known_indeterminate
                ):
                    further_pared.append(
                        (index, result.pared_graph, result.certificate)
                    )
                    reason = "reduces to known indeterminate graph"
                else:
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

    if save:
        if start is None and end is None:
            _save_indeterminate(order, further_graphs, data_dir, replace=True)
        else:
            _update_processed_indeterminate(
                order, processed_indices, further_graphs, data_dir
            )


def main():  # pragma: no cover
    """Entry point for the ``clique-behavior`` console script."""
    _main(sys.argv[1:])


# ---------------------------------------------------------------------------
# Second-stage CLI: optional [LNP06, Theorem 2.6] pass over fast-pass INDETERMINATE
# graphs. See classify_clique_behavior_with_clockwork_pair_search for the classifier
# orchestration; this section only handles CLI parsing, dataset iteration,
# and reporting.
# ---------------------------------------------------------------------------


def _run_fast_pass_indeterminate(
    order: int,
    *,
    data_dir: Path,
    start: int | None,
    end: int | None,
    skip_dominated: bool,
    bound: int,
) -> tuple[int, int, list[tuple[int, nx.Graph, Certificate | None]], set[int]]:
    """Run the fast pass over the dataset, returning only INDETERMINATE graphs.

    .. rubric:: Returns

    tuple[int, int, list[tuple[int, nx.Graph, Certificate | None]], set[int]]
        ``(convergent_count, divergent_count, indeterminate, processed_indices)``
        where ``processed_indices`` includes resolved and skipped graphs, and
        ``indeterminate`` holds ``(index, pared_graph, certificate)`` for
        every graph the fast pass left ``INDETERMINATE``.
    """
    records = _connected_records(order)
    convergent = 0
    divergent = 0
    indeterminate: list[tuple[int, nx.Graph, Certificate | None]] = []
    processed_indices: set[int] = set()

    for index, line in records:
        if start is not None and index < start:
            continue
        if end is not None and index > end:
            break
        processed_indices.add(index)
        graph = nx.from_graph6_bytes(line.encode("ascii"))
        if skip_dominated and find_dominated_vertex(graph) is not None:
            continue
        result = rules.classify_clique_behavior(graph, bound=bound)
        if result.verdict is Verdict.CONVERGENT:
            convergent += 1
        elif result.verdict is Verdict.DIVERGENT:
            divergent += 1
        else:
            indeterminate.append((index, result.pared_graph, result.certificate))

    return convergent, divergent, indeterminate, processed_indices


def _parse_clockwork_pair_args(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the [LNP06, Theorem 2.6] second-stage script."""
    parser = argparse.ArgumentParser(
        description=(
            "[LNP06, Theorem 2.6] (coaffination) pass over graphs the fast "
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
            "Clique bound for the K(graph) fallback used by the [LNP06, Theorem 2.6] "
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


def _main_clockwork_pairs(args: list[str]):
    """Run the [LNP06, Theorem 2.6] pass over fast-pass INDETERMINATE graphs."""
    parsed_args = _parse_clockwork_pair_args(args)
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
    if parsed_args.from_indeterminate_file and (
        parsed_args.start is not None or parsed_args.end is not None
    ):
        _logger.error(
            "--start and --end cannot be used with --from-indeterminate-file."
        )
        sys.exit(1)

    candidates: list[tuple[int, nx.Graph, Certificate | None]]
    if parsed_args.from_indeterminate_file:
        candidates = _load_indeterminate_file_entries_with_metadata(order, data_dir)
        processed_indices = {idx for idx, _, _ in candidates}
        fast_convergent = fast_divergent = None
    else:
        _logger.info("Running fast pass to collect INDETERMINATE graphs...")
        (
            fast_convergent,
            fast_divergent,
            candidates,
            processed_indices,
        ) = _run_fast_pass_indeterminate(
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

        for idx, graph, fast_certificate in candidates:
            theorem_result = rules.classify_clockwork_pair_map(
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
                still_indeterminate.append((idx, graph, fast_certificate))
                if verdict_file is not None:
                    verdict_file.write(
                        f"{idx}\tINDETERMINATE\tunresolved by [LNP06, Theorem 2.6]\n"
                    )

    _logger.info("[LNP06, Theorem 2.6]:")
    _logger.info(f"    newly divergent:         {len(newly_divergent)}")
    _logger.info(f"    remaining indeterminate: {len(still_indeterminate)}")

    if parsed_args.save:
        if parsed_args.from_indeterminate_file:
            _save_indeterminate(order, still_indeterminate, data_dir, replace=True)
        elif parsed_args.start is None and parsed_args.end is None:
            _save_indeterminate(order, still_indeterminate, data_dir, replace=True)
        else:
            _update_processed_indeterminate(
                order, processed_indices, still_indeterminate, data_dir
            )


def main_clockwork_pairs():  # pragma: no cover
    """Entry point for the ``small-behavior-clockwork-pairs`` console script."""
    _main_clockwork_pairs(sys.argv[1:])
