"""
This file gives an interface to use graph data from
Brendan McKay's page (http://cs.anu.edu.au/~bdm/data/graphs.html).
Includes all graphs from 5 to 10 vertices and connected graphs from 6 to 10.
"""

import gzip
from collections.abc import Iterator
from importlib import resources

import networkx as nx

# We define the subpackage where the data files actually reside
DATA_PACKAGE = "pyg6data.data"

_dict_all = {
    5: "graph5.g6",
    6: "graph6.g6",
    7: "graph7.g6",
    8: "graph8.g6.gz",
    9: "graph9.g6.gz",
    10: "graph10.g6.gz",
}

_dict_connected = {
    6: "graph6c.g6.gz",
    7: "graph7c.g6.gz",
    8: "graph8c.g6.gz",
    9: "graph9c.g6.gz",
    10: "graph10c.g6.gz",
}


def _get_data_file_path(filename: str) -> resources.abc.Traversable:
    """Helper function to resolve the modern path to a package data file."""
    # resources.files() returns a Traversable object representing the
    # directory.
    # .joinpath() securely targets the specific file.
    return resources.files(DATA_PACKAGE).joinpath(filename)


def graph_generator(n: int, connected: bool = True) -> Iterator[nx.Graph]:
    """
    Yields NetworkX graphs from a g6.gz file.

    Args:
        n (int): Order of the graphs (number of nodes).
        connected (bool): If True, reads connected graphs file; else,
            reads all graphs file. Defaults to True.

    Yields:
        nx.Graph: A NetworkX graph read from the file.

    Raises:
        ValueError: If the requested order 'n' is not available.
    """
    the_dict = _dict_connected if connected else _dict_all

    if n not in the_dict:
        msg = f"Data for n={n} (connected={connected}) is not available."
        raise ValueError(msg)

    # Get the secure path to the file inside the installed package
    filename = the_dict[n]
    file_path = _get_data_file_path(filename)

    if filename.endswith(".gz"):
        with file_path.open("rb") as raw_file:
            with gzip.open(raw_file, "rt", encoding="utf-8") as graph_file:
                for graph_string in graph_file:
                    graph_string = graph_string.strip()
                    if graph_string:
                        yield nx.from_graph6_bytes(graph_string.encode("utf-8"))
    else:
        with file_path.open("r", encoding="utf-8") as graph_file:
            for graph_string in graph_file:
                graph_string = graph_string.strip()
                if graph_string:
                    yield nx.from_graph6_bytes(graph_string.encode("utf-8"))


def list_graphs(n: int, connected: bool = True) -> list[nx.Graph]:
    """List of graphs of a given order, from B. McKay data."""
    return list(graph_generator(n, connected))


def small_torsion_graphs() -> list[nx.Graph]:
    """Loads the small-torsion graphs."""
    file_path = _get_data_file_path("small-torsion.g6")
    list_of_graphs = []
    # This is a regular .g6 file (not gzipped)
    with file_path.open("r", encoding="utf-8") as graph_file:
        for graph_string in graph_file:
            graph_string = graph_string.strip()
            if graph_string:
                list_of_graphs.append(
                    nx.from_graph6_bytes(graph_string.encode("utf-8"))
                )
    return list_of_graphs
