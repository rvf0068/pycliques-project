"""Classify graphs supplied as graph6 strings or files."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import networkx as nx
from rich.console import Console
from rich.logging import RichHandler

from pycliques import __version__
from pycliques.small import classify_clique_behavior

_logger = logging.getLogger(__name__)


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the classify-behavior script."""
    parser = argparse.ArgumentParser(
        description="Classify graphs supplied as graph6 strings or files."
    )
    parser.add_argument(
        "--version", action="version", version=f"pycliques {__version__}"
    )
    parser.add_argument(
        "input",
        help="A graph6 string or a file containing one graph6 string per line",
        metavar="GRAPH6_OR_FILE",
    )
    parser.add_argument(
        "--bound",
        type=int,
        default=30,
        help="Maximum number of cliques allowed at each iteration (default: 30)",
        metavar="INT",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show classification progress (default: True)",
    )
    return parser.parse_args(args)


def _graph6_lines(input_value: str) -> Iterator[str]:
    """Yield non-empty graph6 lines from a graph6 string or existing file."""
    path = Path(input_value)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        with path.open(encoding="ascii") as graph_file:
            yield from (line.strip() for line in graph_file if line.strip())
    else:
        yield input_value.strip()


def _classify_graphs(
    graph6_lines: Iterable[str], bound: int, progress: bool
) -> None:
    """Classify graph6 lines and print one result per graph."""
    for index, graph6 in enumerate(graph6_lines, start=1):
        if progress:
            _logger.info(f"Classifying graph {index}...")
        graph = nx.from_graph6_bytes(graph6.encode("ascii"))
        on_iterate = (
            lambda iteration: _logger.info(f"Working with iterate {iteration}")
            if progress
            else None
        )
        result = classify_clique_behavior(
            graph, bound=bound, on_iterate=on_iterate
        )
        print(f"{index}\t{result.verdict.name}\t{result.reason}")


def _setup_logging() -> None:
    """Configure Rich logging for the CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(stderr=True))],
    )


def _main(args: list[str]) -> None:
    """Classify graphs supplied on the command line."""
    parsed_args = _parse_args(args)
    _setup_logging()
    _classify_graphs(
        _graph6_lines(parsed_args.input), parsed_args.bound, parsed_args.progress
    )
    if parsed_args.progress:
        _logger.info(f"Done at {datetime.now():%H:%M:%S}")


def main() -> None:  # pragma: no cover
    """Entry point for the ``classify-behavior`` console script."""
    _main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
