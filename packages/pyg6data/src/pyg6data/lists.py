"""
This file gives an interface to use graph data from
Brendan McKay's page (http://cs.anu.edu.au/~bdm/data/graphs.html).
Includes all graphs from 5 to 10 vertices and connected graphs from 6 to 10.

Digraph data (non-isomorphic directed graphs on 1 to 6 vertices) is from
Brendan McKay's page (https://users.cecs.anu.edu.au/~bdm/data/digraphs.html).
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

# Non-isomorphic directed graphs obtained from Brendan McKay's page:
# https://users.cecs.anu.edu.au/~bdm/data/digraphs.html
_dict_digraphs = {
    1: "dig1.d6",
    2: "dig2.d6",
    3: "dig3.d6",
    4: "dig4.d6",
    5: "dig5.d6",
    6: "dig6.d6.gz",
}

# Cubic connected graphs obtained from the House of Graphs database:
# https://houseofgraphs.org/
# Citation: K. Coolsaet, S. D'hondt and J. Goedgebeur, House of Graphs 2.0:
# A database of interesting graphs and more, Discrete Applied Mathematics,
# 325:97-107, 2023. Available at https://houseofgraphs.org
_dict_cubic = {
    8: "cub08.g6",
    10: "cub10.g6",
    12: "cub12.g6",
    14: "cub14.g6",
    16: "cub16.g6",
    18: "cub18.g6.gz",
    20: "cub20.g6.gz",
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


def cubic_graph_generator(n: int) -> Iterator[nx.Graph]:
    """Yields cubic connected graphs of a given order from House of Graphs data.

    Data obtained from the House of Graphs database
    (https://houseofgraphs.org/).

    .. rubric:: Parameters

    n : int
        Order of the graphs (number of nodes). Must be an even number
        between 8 and 20.

    .. rubric:: Yields

    networkx.Graph
        A cubic connected NetworkX graph of order ``n``.

    .. rubric:: Raises

    ValueError
        If the requested order ``n`` is not available.

    .. rubric:: Notes

    Citation: K. Coolsaet, S. D'hondt and J. Goedgebeur, House of Graphs
    2.0: A database of interesting graphs and more, *Discrete Applied
    Mathematics*, 325:97-107, 2023. Available at https://houseofgraphs.org
    """
    if n not in _dict_cubic:
        msg = f"Cubic graph data for n={n} is not available."
        raise ValueError(msg)

    filename = _dict_cubic[n]
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


def list_cubic_graphs(n: int) -> list[nx.Graph]:
    """List of cubic connected graphs of a given order, from House of Graphs data.

    .. rubric:: Parameters

    n : int
        Order of the graphs (number of nodes). Must be an even number
        between 8 and 20.

    .. rubric:: Returns

    list[networkx.Graph]
        All cubic connected graphs of order ``n``.

    .. rubric:: Notes

    Citation: K. Coolsaet, S. D'hondt and J. Goedgebeur, House of Graphs
    2.0: A database of interesting graphs and more, *Discrete Applied
    Mathematics*, 325:97-107, 2023. Available at https://houseofgraphs.org
    """
    return list(cubic_graph_generator(n))


def parse_digraph6(d6_str: str) -> nx.DiGraph:
    """Parse a digraph6 string into a NetworkX DiGraph.

    Digraph6 is a compact ASCII encoding for directed graphs defined by
    Brendan McKay.  Each string encodes the number of vertices followed
    by the adjacency matrix in row-major order using 6-bit ASCII
    characters.

    .. rubric:: Parameters

    d6_str : str
        A digraph6 string, optionally prefixed with ``>>digraph6<<``.
        Valid strings must start with ``&`` (after the optional prefix).

    .. rubric:: Returns

    networkx.DiGraph
        The directed graph encoded by *d6_str*.

    .. rubric:: Raises

    ValueError
        If *d6_str* does not start with ``&`` (after stripping the
        optional ``>>digraph6<<`` header), or if the string is too short
        to represent all edges.

    .. rubric:: Examples

    >>> from pyg6data import parse_digraph6
    >>> g = parse_digraph6("&A?")
    >>> g.number_of_nodes(), g.number_of_edges()
    (2, 0)
    >>> g = parse_digraph6("&AO")
    >>> list(g.edges())
    [(0, 1)]
    """
    d6_str = d6_str.strip()
    if d6_str.startswith(">>digraph6<<"):
        d6_str = d6_str[12:]

    if not d6_str.startswith("&"):
        raise ValueError("Valid digraph6 strings must start with '&'")
    d6_str = d6_str[1:]

    data = [ord(c) - 63 for c in d6_str]

    if data[0] <= 62:
        n = data[0]
        data = data[1:]
    elif data[1] <= 62:
        n = (data[1] << 12) | (data[2] << 6) | data[3]
        data = data[4:]
    else:
        n = (
            (data[2] << 30)
            | (data[3] << 24)
            | (data[4] << 18)
            | (data[5] << 12)
            | (data[6] << 6)
            | data[7]
        )
        data = data[8:]

    bits = [int(b) for val in data for b in format(val, "06b")]

    if len(bits) < n * n:
        raise ValueError("String is too short to represent all edges")

    g: nx.DiGraph = nx.DiGraph()
    g.add_nodes_from(range(n))

    idx = 0
    for i in range(n):
        for j in range(n):
            if bits[idx] == 1:
                g.add_edge(i, j)
            idx += 1

    return g


def digraph_generator(n: int) -> Iterator[nx.DiGraph]:
    """Yield all non-isomorphic directed graphs on *n* vertices.

    Data obtained from Brendan McKay's digraph data page
    (https://users.cecs.anu.edu.au/~bdm/data/digraphs.html).

    .. rubric:: Parameters

    n : int
        Number of vertices.  Available values: 1 to 6.

    .. rubric:: Yields

    networkx.DiGraph
        A directed graph with ``n`` vertices.

    .. rubric:: Raises

    ValueError
        If ``n`` is not in the available range (1--6).

    .. rubric:: Examples

    >>> from pyg6data import digraph_generator
    >>> gen = digraph_generator(1)
    >>> g = next(gen)
    >>> g.number_of_nodes(), g.number_of_edges()
    (1, 0)
    >>> sum(1 for _ in digraph_generator(3))
    16
    """
    if n not in _dict_digraphs:
        msg = f"Digraph data for n={n} is not available."
        raise ValueError(msg)

    filename = _dict_digraphs[n]
    file_path = _get_data_file_path(filename)

    if filename.endswith(".gz"):
        with file_path.open("rb") as raw_file:
            with gzip.open(raw_file, "rt", encoding="utf-8") as graph_file:
                for line in graph_file:
                    line = line.strip()
                    if line:
                        yield parse_digraph6(line)
    else:
        with file_path.open("r", encoding="utf-8") as graph_file:
            for line in graph_file:
                line = line.strip()
                if line:
                    yield parse_digraph6(line)


def list_digraphs(n: int) -> list[nx.DiGraph]:
    """List all non-isomorphic directed graphs on *n* vertices.

    Data obtained from Brendan McKay's digraph data page
    (https://users.cecs.anu.edu.au/~bdm/data/digraphs.html).

    .. rubric:: Parameters

    n : int
        Number of vertices.  Available values: 1 to 6.

    .. rubric:: Returns

    list[networkx.DiGraph]
        All non-isomorphic directed graphs on ``n`` vertices.

    .. rubric:: Examples

    >>> from pyg6data import list_digraphs
    >>> len(list_digraphs(2))
    3
    >>> len(list_digraphs(4))
    218
    """
    return list(digraph_generator(n))
