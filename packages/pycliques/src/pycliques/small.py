"""Determine the clique behavior of small graphs."""

from __future__ import annotations

import argparse
import gzip
import logging
import sys

import networkx as nx
from pyg6data.lists import _dict_connected, _get_data_file_path
from rich.logging import RichHandler

from pycliques import __version__
from pycliques.cliques import clique_graph
from pycliques.dominated import completely_pared_graph
from pycliques.helly import is_clique_helly
from pycliques.named import complement_of_cycle, suspension_of_cycle
from pycliques.retractions import retracts, special_octahedra

_logger = logging.getLogger(__name__)


def eventually_retracts_specially(graph: nx.Graph, max_steps: int = 15) -> bool:
    """Check if iterated clique graphs eventually contain a special octahedron.

    Starting from ``graph``, repeatedly compute the completely-pared clique
    graph.  Return ``True`` as soon as one iterate contains a special
    octahedron (see :func:`pycliques.retractions.special_octahedra`).

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    max_steps : int
        Maximum number of clique-graph iterations (default 15).

    .. rubric:: Returns

    bool
        ``True`` if a special octahedron is found within ``max_steps``
        iterations.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.small import eventually_retracts_specially
    >>> eventually_retracts_specially(nx.cycle_graph(4))
    False
    """
    g_curr = graph
    for _ in range(max_steps):
        # We pare the clique graph to keep it small
        g_curr = completely_pared_graph(clique_graph(g_curr))

        # If it collapses to a point (or empty), it converged safely
        if g_curr.order() <= 1:
            return False

        if special_octahedra(g_curr):
            return True

    return False


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

    if order not in _dict_connected:
        _logger.error(f"Error: Internal data for order {order} not available.")
        sys.exit(1)

    # 1. Precompute expensive targets ONCE
    _logger.info("Precomputing target mathematical structures...")
    sc5 = suspension_of_cycle(5)
    sc6 = suspension_of_cycle(6)
    cc8 = complement_of_cycle(8)

    convergent = []
    divergent = []
    further = []

    _logger.info(f"Beginning analysis of connected graphs of order {order}...")

    # 2. Securely resolve the dataset path
    data_path = _get_data_file_path(_dict_connected[order])

    with data_path.open("rb") as raw_file:
        with gzip.open(raw_file, "rt", encoding="utf-8") as graph_file:
            for index, line in enumerate(graph_file):
                assert isinstance(line, str)
                graph = nx.from_graph6_bytes(bytes(line.strip(), "utf-8"))
                behavior = ""

                # -------------------------------------------------------------
                # The Classification Pipeline
                # -------------------------------------------------------------
                if is_clique_helly(graph):
                    behavior = "is Helly"
                    convergent.append(index)

                elif is_clique_helly(clique_graph(graph)):
                    behavior = "is eventually Helly"
                    convergent.append(index)

                elif special_octahedra(graph):
                    behavior = "has an induced special octahedron"
                    divergent.append(index)

                elif retracts(graph, sc5):
                    behavior = "retracts to Susp(C_5)"
                    divergent.append(index)

                elif retracts(graph, sc6):
                    behavior = "retracts to Susp(C_6)"
                    divergent.append(index)

                elif retracts(graph, cc8):
                    behavior = "retracts to Comp(C_8)"
                    divergent.append(index)

                elif eventually_retracts_specially(graph):
                    behavior = "eventually has a special octahedron"
                    divergent.append(index)

                else:
                    behavior = "has character unknown so far"
                    further.append(index)

                _logger.debug(f"Graph {index}: {behavior}")

                # Progress tracker for massive files
                if index > 0 and index % 10000 == 0:
                    _logger.info(f"Processed {index} graphs...")

    # For edge cases where the file was empty
    total_processed = index + 1 if "index" in locals() else 0

    _logger.info(f"Analysis Complete! Processed {total_processed} total graphs.")
    _logger.info(f"Indices that deserve further study: {further}")
    _logger.info(f"Total convergent graphs: {len(convergent)}")
    _logger.info(f"Total divergent graphs: {len(divergent)}")
    _logger.info(f"Total unknown graphs (further study): {len(further)}")


def main():
    """Entry point for the ``clique-behavior`` console script."""
    _main(sys.argv[1:])


if __name__ == "__main__":
    main()
