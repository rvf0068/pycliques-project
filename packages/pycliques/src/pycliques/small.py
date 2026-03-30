"""Determine the clique behavior of small graphs."""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

import networkx as nx
from pyg6data.lists import _dict_connected, _get_data_file_path
from rich.logging import RichHandler

from pycliques import __version__
from pycliques.cliques import clique_graph
from pycliques.clockwork import is_clique_divergent_clockwork, recognize_clockwork
from pycliques.dominated import completely_pared_graph
from pycliques.helly import is_clique_helly
from pycliques.named import complement_of_cycle, suspension_of_cycle
from pycliques.retractions import retracts, special_octahedra

_logger = logging.getLogger(__name__)

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
    """
    path = _indeterminate_file_path(order, data_dir)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            f"# Indeterminate clique behavior – connected graphs of order {order}\n"
        )
        f.write("# Format: original_index pared_order graph6\n")
        for idx, graph in indeterminate:
            g = nx.convert_node_labels_to_integers(graph)
            g6 = nx.to_graph6_bytes(g, header=False).decode("ascii").strip()
            f.write(f"{idx} {g.order()} {g6}\n")
    _logger.info(f"Saved {len(indeterminate)} indeterminate graphs to {path}")


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
    graph.  Return ``True`` as soon as one iterate is clique-Helly.

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
    graph.  Return ``True`` as soon as one iterate contains a special
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
        if special_octahedra(g_curr):
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

    if order not in _dict_connected:
        _logger.error(f"Error: Internal data for order {order} not available.")
        sys.exit(1)

    # 1. Precompute expensive targets ONCE
    _logger.info("Precomputing target mathematical structures...")
    sc5 = suspension_of_cycle(5)
    sc6 = suspension_of_cycle(6)
    cc8 = complement_of_cycle(8)

    # Load previously saved indeterminate graphs for smaller orders
    known_indeterminate: dict[int, list[nx.Graph]] = {}
    if lookup:
        known_indeterminate = _load_indeterminate_graphs(order, data_dir)

    convergent = []
    divergent = []
    further = []
    further_pared: list[tuple[int, nx.Graph]] = []
    further_graphs: list[tuple[int, nx.Graph]] = []

    _logger.info(f"Beginning analysis of connected graphs of order {order}...")

    # 2. Securely resolve the dataset path
    data_path = _get_data_file_path(_dict_connected[order])

    with data_path.open("rb") as raw_file:
        with gzip.open(raw_file, "rt", encoding="utf-8") as graph_file:
            for index, line in enumerate(graph_file):
                assert isinstance(line, str)
                graph = nx.from_graph6_bytes(bytes(line.strip(), "utf-8"))
                behavior = ""
                graph = completely_pared_graph(graph)

                if known_indeterminate and _is_known_indeterminate(
                    graph, known_indeterminate
                ):
                    behavior = "reduces to known indeterminate graph"
                    further_pared.append((index, graph))

                elif is_eventually_helly(graph):
                    behavior = "is eventually Helly"
                    convergent.append(index)

                elif recognize_clockwork(graph)[0]:
                    if is_clique_divergent_clockwork(graph):
                        behavior = "is clockwork divergent"
                        divergent.append(index)
                    else:
                        behavior = "is clockwork convergent"
                        convergent.append(index)

                elif recognize_clockwork(clique_graph(graph))[0]:
                    if is_clique_divergent_clockwork(clique_graph(graph)):
                        behavior = "is clockwork divergent"
                        divergent.append(index)
                    else:
                        behavior = "is clockwork convergent"
                        convergent.append(index)

                elif eventually_retracts_specially(graph):
                    behavior = "eventually has a special octahedron"
                    divergent.append(index)

                elif retracts(graph, sc5):  # pragma: no cover
                    behavior = "retracts to Susp(C_5)"
                    divergent.append(index)

                elif retracts(graph, sc6):  # pragma: no cover
                    behavior = "retracts to Susp(C_6)"
                    divergent.append(index)

                elif retracts(graph, cc8):  # pragma: no cover
                    behavior = "retracts to Comp(C_8)"
                    divergent.append(index)

                else:  # pragma: no cover
                    behavior = "has character unknown so far"
                    further.append(index)
                    further_graphs.append((index, graph))

                _logger.debug(f"Graph {index}: {behavior}")

                # Progress tracker for massive files
                if index > 0 and index % 10000 == 0:  # pragma: no cover
                    _logger.info(f"Processed {index} graphs...")

    # For edge cases where the file was empty
    total_processed = index + 1 if "index" in locals() else 0

    _logger.info(f"Analysis Complete! Processed {total_processed} total graphs.")
    _logger.info(f"Indices that deserve further study: {further}")
    _logger.info(f"Total convergent graphs: {len(convergent)}")
    _logger.info(f"Total divergent graphs: {len(divergent)}")
    _logger.info(f"Total graphs reduced to unknown: {len(further_pared)}")
    _logger.info(f"Total unknown graphs (further study): {len(further)}")

    if save and further_graphs:
        _save_indeterminate(order, further_graphs, data_dir)


def main():  # pragma: no cover
    """Entry point for the ``clique-behavior`` console script."""
    _main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
