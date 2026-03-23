"""Graph retractions and induced subgraphs."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterator
from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism

from .cliques import clique_graph
from .coaffinations import automorphisms
from .dominated import closed_neighborhood, completely_pared_graph
from .named import complement_of_cycle, octahedron, suspension_of_cycle

# ---------------------------------------------------------------------------
# Logging Setup: The "NullHandler" prevents logs from polluting doctests
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def dict_to_tuple(the_dict: dict) -> tuple:
    """Convert a dictionary to a sorted tuple of key-value pairs.

    .. rubric:: Parameters

    the_dict : dict
        Dictionary to convert.

    .. rubric:: Returns

    tuple
        Tuple of ``(key, value)`` pairs.

    .. rubric:: Examples

    >>> from pycliques.retractions import dict_to_tuple
    >>> dict_to_tuple({1: 'a', 2: 'b'})
    ((1, 'a'), (2, 'b'))
    """
    # .items() already yields (key, value) pairs natively
    return tuple(the_dict.items())


def invert_dict(the_dict: dict) -> dict:
    """Invert a dictionary's keys and values.

    Assumes the mapping is bijective (injective).

    .. rubric:: Parameters

    the_dict : dict
        Dictionary whose keys and values will be swapped.

    .. rubric:: Returns

    dict
        A new dictionary with keys and values exchanged.

    .. rubric:: Examples

    >>> from pycliques.retractions import invert_dict
    >>> invert_dict({0: 'a', 1: 'b'})
    {'a': 0, 'b': 1}
    """
    # Dictionary comprehension is the fastest way to build a dict in Python
    return {v: k for k, v in the_dict.items()}


def graph_from_gap_adjacency_list(the_list: list[list[int]]) -> nx.Graph:
    """Create a 0-indexed graph from a 1-indexed GAP adjacency list.

    .. rubric:: Parameters

    the_list : list[list[int]]
        Adjacency list where vertex indices start at 1 (GAP convention).

    .. rubric:: Returns

    networkx.Graph
        Graph with 0-indexed vertices.

    .. rubric:: Examples

    >>> from pycliques.retractions import graph_from_gap_adjacency_list
    >>> g = graph_from_gap_adjacency_list([[2], [1]])
    >>> sorted(g.edges())
    [(0, 1)]
    """
    graph = nx.Graph()
    for i, adj in enumerate(the_list):
        graph.add_edges_from([(i, v - 1) for v in adj])
    return graph


def is_map(domain: nx.Graph, codomain: nx.Graph, ismap: dict) -> bool:
    """Determine whether ``ismap`` defines a graph homomorphism.

    It is not required that every vertex in ``domain`` has a value
    determined in ``ismap``.

    .. rubric:: Parameters

    domain : networkx.Graph
        Source graph.
    codomain : networkx.Graph
        Target graph.
    ismap : dict
        Partial or total vertex mapping from ``domain`` to ``codomain``.

    .. rubric:: Returns

    bool
        ``True`` if the mapping preserves adjacency.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import is_map
    >>> mapping = {0: 0, 1: 1, 2: 0, 3: 1}
    >>> is_map(nx.cycle_graph(4), nx.complete_graph(2), mapping)
    True
    >>> mapping = {0: 0, 1: 1}
    >>> is_map(nx.cycle_graph(4), nx.complete_graph(2), mapping)
    True
    """
    for u, v in domain.edges():
        if u in ismap and v in ismap:
            wu, wv = ismap[u], ismap[v]
            # Must map to an edge, or collapse to a single vertex
            if wu != wv and not codomain.has_edge(wu, wv):
                return False
    return True


def _extension_of_map(large: nx.Graph, small: nx.Graph, mapp: dict, v: Any) -> set:
    """Find the set of vertices of `small` that could be images of `v`."""
    common = set(small.nodes())
    for w in set(large[v]).intersection(mapp.keys()):
        common &= closed_neighborhood(small, mapp[w])
    return common


def _extend_retraction(
    large: nx.Graph, small: nx.Graph, state: tuple
) -> Iterator[tuple]:
    """Generator to backtrack and complete a retraction from large to small."""
    ret = dict(state)
    remaining = list(set(large.nodes()) - set(ret.keys()))

    for v in remaining:
        for w in _extension_of_map(large, small, ret, v):
            if len(ret) == large.order() - 1:
                yield ((v, w),)
            else:
                for res in _extend_retraction(large, small, state + ((v, w),)):
                    yield ((v, w),) + res


def retraction(large: nx.Graph, small: nx.Graph) -> Iterator[tuple[dict, dict]]:
    """Yield all retractions from ``large`` onto ``small``.

    Each retraction is a pair of dictionaries:

    1. A surjective map from ``large`` to ``small`` (the retraction).
    2. A map from ``small`` to ``large`` (the inclusion).

    .. rubric:: Parameters

    large : networkx.Graph
        Graph that may retract.
    small : networkx.Graph
        Target subgraph.

    .. rubric:: Yields

    tuple[dict, dict]
        ``(retraction_map, inclusion_map)`` pairs.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import retraction
    >>> list(retraction(nx.wheel_graph(4), nx.cycle_graph(4)))
    []
    >>> list(retraction(nx.path_graph(3), nx.path_graph(2)))
    [({0: 0, 1: 1, 2: 0}, {0: 0, 1: 1}), ({0: 0, 1: 1, 2: 1}, {0: 0, 1: 1})]
    """
    GM = isomorphism.GraphMatcher(large, small)
    rets = GM.subgraph_isomorphisms_iter()

    a_small = list(automorphisms(small))
    a_large = list(automorphisms(large))

    # PERFORMANCE FIX: Use a set of frozensets for O(1) mathematical symmetry checks
    seen_signatures = set()

    for ret in rets:
        ret_sig = frozenset(ret.items())

        if ret_sig not in seen_signatures:
            if large.order() == small.order():
                yield (ret, invert_dict(ret))
            else:
                state = tuple(ret.items())
                for ext in _extend_retraction(large, small, state):
                    yield (dict(state + ext), invert_dict(ret))

            # Add all symmetric equivalent inclusions to the seen set
            for auto_s in a_small:
                for auto_l in a_large:
                    sym_sig = frozenset((auto_l[x], auto_s[ret[x]]) for x in ret)
                    seen_signatures.add(sym_sig)


def retracts(large: nx.Graph, small: nx.Graph) -> tuple[dict, dict] | None:
    """Return a retraction from ``large`` to ``small`` if one exists.

    .. rubric:: Parameters

    large : networkx.Graph
        Graph that may retract.
    small : networkx.Graph
        Target subgraph.

    .. rubric:: Returns

    tuple[dict, dict] | None
        The first retraction found, or ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import retracts
    >>> retracts(nx.path_graph(3), nx.path_graph(2)) is not None
    True
    >>> retracts(nx.wheel_graph(4), nx.cycle_graph(4)) is None
    True
    """
    try:
        return next(retraction(large, small))
    except StopIteration:
        return None


def retracts_to(subgraph: nx.Graph) -> Callable[[nx.Graph], tuple[dict, dict] | None]:
    """Return a function that checks retraction to ``subgraph``.

    .. rubric:: Parameters

    subgraph : networkx.Graph
        Fixed target subgraph.

    .. rubric:: Returns

    Callable
        A function ``f(g)`` returning the retraction or ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import retracts_to
    >>> checker = retracts_to(nx.path_graph(2))
    >>> checker(nx.path_graph(3)) is not None
    True
    """
    return lambda g: retracts(g, subgraph)


def has_induced(large: nx.Graph, small: nx.Graph) -> dict | None:
    """Return an injective map witnessing ``small`` as an induced subgraph.

    .. rubric:: Parameters

    large : networkx.Graph
        Host graph.
    small : networkx.Graph
        Pattern graph.

    .. rubric:: Returns

    dict | None
        A node mapping from ``large`` to ``small`` if one exists,
        otherwise ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import has_induced
    >>> has_induced(nx.complete_graph(4), nx.path_graph(3)) is not None
    True
    >>> has_induced(nx.cycle_graph(4), nx.complete_graph(3)) is None
    True
    """
    GM = isomorphism.GraphMatcher(large, small)
    try:
        result: dict = next(GM.subgraph_isomorphisms_iter())
        return result
    except StopIteration:
        return None


# ---------------------------------------------------------------------------
# CLI Application Logic
# ---------------------------------------------------------------------------


def _string_to_graph(string: str) -> nx.Graph:
    if string.startswith("sc"):
        return suspension_of_cycle(int(string[2:]))
    elif string.startswith("cc"):
        return complement_of_cycle(int(string[2:]))
    elif string.startswith("o"):
        return octahedron(int(string[1:]))
    raise ValueError(f"Unknown graph string format: {string}")


def _parse_args(args: list[str]) -> argparse.Namespace:
    from pycliques import __version__

    parser = argparse.ArgumentParser(description="Retractions to octahedra")
    parser.add_argument(
        "--version", action="version", version=f"pycliques {__version__}"
    )
    parser.add_argument(dest="n", help="index of clique graph", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        dest="large", help="large graph in g6 format", type=str, metavar="STR"
    )
    parser.add_argument(
        dest="small", help="small graph in g6 format", type=str, metavar="STR"
    )
    return parser.parse_args(args)


def _setup_logging(loglevel: int | None):
    """Safely setup logging ONLY when running as a CLI script."""
    if loglevel is not None:
        logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
        logging.basicConfig(
            level=loglevel,
            stream=sys.stdout,
            format=logformat,
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _main(args: list[str]):
    args_parsed = _parse_args(args)
    _setup_logging(args_parsed.loglevel)

    _logger.info("Parsing input graphs...")
    large = nx.from_graph6_bytes(bytes(args_parsed.large, "utf8"))
    small = _string_to_graph(args_parsed.small)

    for i in range(args_parsed.n):
        _logger.info(f"Iterating the clique operator (Step {i + 1})")
        large = completely_pared_graph(clique_graph(large))

    large = nx.convert_node_labels_to_integers(large)
    _logger.info(f"The large graph has order {large.order()}")
    _logger.info("Searching for retractions...")

    has_retraction = retracts(large, small)
    if has_retraction:
        print(f"Found {has_retraction}")
    else:
        print("Sorry, could not find it!")

    _logger.info("Script ends here")
