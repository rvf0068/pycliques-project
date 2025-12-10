from __future__ import annotations

import networkx as nx


def is_regular(graph: nx.Graph, k: int) -> bool:
    """Return True when every vertex has degree ``k``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.surfaces import is_regular
    >>> is_regular(nx.cycle_graph(4), 2)
    True
    >>> is_regular(nx.path_graph(3), 2)
    False
    """

    if len(graph) == 0:
        return True
    return all(graph.degree(v) == k for v in graph)


def is_cycle(graph: nx.Graph) -> bool:
    """Return True if the graph is a single cycle.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.surfaces import is_cycle
    >>> is_cycle(nx.cycle_graph(5))
    True
    >>> is_cycle(nx.path_graph(5))
    False
    """

    if len(graph) == 0:
        return False
    return nx.is_connected(graph) and is_regular(graph, 2)


def is_path(graph: nx.Graph) -> bool:
    """Return True if the graph is a simple path.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.surfaces import is_path
    >>> is_path(nx.path_graph(4))
    True
    >>> is_path(nx.cycle_graph(4))
    False
    """

    if len(graph) == 0:
        return False
    leaves = [x for x in graph if graph.degree(x) == 1]
    return nx.is_tree(graph) and len(leaves) == 2


def open_neighborhood(graph: nx.Graph, v: int) -> nx.Graph:
    """Return the subgraph induced by the neighbors of ``v``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.surfaces import open_neighborhood
    >>> G = nx.cycle_graph(4)
    >>> open_neighborhood(G, 0).nodes()
    NodeView((1, 3))
    """

    return graph.subgraph(graph[v]).copy()


def is_closed_surface(graph: nx.Graph) -> bool:
    """Return True if the open neighborhood of any vertex is a 'big cycle'.

    A 'big cycle' has length at least 4.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.surfaces import is_closed_surface
    >>> is_closed_surface(nx.octahedral_graph())
    True
    >>> is_closed_surface(nx.cycle_graph(4))
    False
    """

    for v in graph:
        on = open_neighborhood(graph, v)
        if on.order() < 4 or not is_cycle(on):
            return False
    return True


def is_surface(graph: nx.Graph) -> bool:
    """Return True if every vertex sees either a cycle or path neighborhood.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.surfaces import is_surface
    >>> is_surface(nx.diamond_graph())
    True
    >>> is_surface(nx.cycle_graph(4))
    False
    """

    for v in graph:
        on = open_neighborhood(graph, v)
        if on.order() < 2 or (not is_cycle(on) and not is_path(on)):
            return False
    return True
