from __future__ import annotations
import networkx as nx


def closed_neighborhood(graph: nx.Graph, v) -> set:
    """Return the closed neighborhood of ``v`` (neighbors plus the vertex).

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import closed_neighborhood
    >>> closed_neighborhood(nx.path_graph(4), 0)
    {0, 1}
    """
    return set(graph[v]) | {v}


def dominates(graph: nx.Graph, v, w) -> bool:
    """Return True when ``v`` dominates ``w``.

    Vertex ``v`` dominates ``w`` if the closed neighborhood of ``w`` is a
    subset of the closed neighborhood of ``v``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import dominates
    >>> path = nx.path_graph(4)
    >>> dominates(path, 1, 0)
    True
    >>> dominates(path, 0, 1)
    False
    """
    neigh_v = closed_neighborhood(graph, v)
    neigh_w = closed_neighborhood(graph, w)
    return neigh_w.issubset(neigh_v)


def is_dominated_vertex(
    graph: nx.Graph, v, return_dominator: bool = False
) -> bool | tuple[bool, int]:
    """Return whether ``v`` is dominated by some other vertex.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import is_dominated_vertex
    >>> path = nx.path_graph(4)
    >>> is_dominated_vertex(path, 0)
    True
    >>> is_dominated_vertex(path, 0, return_dominator=True)
    (True, 1)
    >>> is_dominated_vertex(path, 2)
    False
    """
    for u in graph:
        if u != v:
            if dominates(graph, u, v):
                if return_dominator:
                    return (True, u)
                else:
                    return True
    return False


def find_dominated_vertex(graph: nx.Graph) -> int | None:
    """Return the first dominated vertex in ``graph`` if one exists.

    The result may be ``0``, so prefer identity comparisons when checking for
    ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import find_dominated_vertex
    >>> find_dominated_vertex(nx.path_graph(4))
    0
    >>> find_dominated_vertex(nx.cycle_graph(4)) is None
    True
    """
    for v in graph:
        if is_dominated_vertex(graph, v):
            return v
    return None


def remove_dominated_vertex(graph: nx.Graph) -> nx.Graph:
    """Return a copy of ``graph`` with one dominated vertex removed.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import remove_dominated_vertex
    >>> sorted(remove_dominated_vertex(nx.path_graph(4)).nodes())
    [1, 2, 3]
    >>> sorted(remove_dominated_vertex(nx.cycle_graph(4)).nodes())
    [0, 1, 2, 3]
    """
    g1 = graph.copy()
    x = find_dominated_vertex(graph)

    # SAFETY: Explicit check handles vertex '0' correctly
    if x is None:
        return g1
    else:
        g1.remove_node(x)
        return g1


def twin_classes(graph: nx.Graph) -> list[list]:
    """Group vertices that share the same closed neighborhood.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import twin_classes
    >>> graph = nx.complete_graph(3)
    >>> graph.add_node(3)
    >>> sorted(sorted(cls) for cls in twin_classes(graph))
    [[0, 1, 2], [3]]
    """
    neighborhoods = {}
    for v in graph:
        # Create a hashable signature for the neighborhood (sorted tuple)
        nb_signature = tuple(sorted(closed_neighborhood(graph, v)))

        if nb_signature not in neighborhoods:
            neighborhoods[nb_signature] = []
        neighborhoods[nb_signature].append(v)

    return list(neighborhoods.values())


def pared_graph(graph: nx.Graph) -> nx.Graph:
    """Return the pared graph (one vertex per twin class, no dominated nodes).

    The operation keeps a representative from each twin class and removes any
    representative that is strictly dominated by another vertex outside its
    class.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import pared_graph
    >>> sorted(pared_graph(nx.path_graph(4)).nodes())
    [1, 2]
    """
    twins = twin_classes(graph)

    # Identify representatives
    reps = [aclass[0] for aclass in twins]

    nodes_to_keep = []
    for vertex in reps:
        # We assume is_dominated_vertex checks against ALL nodes.
        if not is_dominated_vertex(graph, vertex):
            nodes_to_keep.append(vertex)
        else:
            # If dominated, check if it's only dominated by its own twins
            dom_info = is_dominated_vertex(graph, vertex,
                                           return_dominator=True)
            if dom_info:
                dominator = dom_info[1]
                neigh_v = closed_neighborhood(graph, vertex)
                neigh_dom = closed_neighborhood(graph, dominator)
                # If neighborhoods match, they are twins -> Keep rep
                if neigh_v == neigh_dom:
                    nodes_to_keep.append(vertex)
            else:
                nodes_to_keep.append(vertex)

    return graph.subgraph(nodes_to_keep).copy()


def pared_index(
        graph: nx.Graph,
        return_cp: bool = False
) -> int | tuple[int, nx.Graph]:
    """Return how many pared-graph iterations are needed until stability.

    When ``return_cp`` is True, the pared index and the final fixed-point
    graph are returned.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import pared_index
    >>> pared_index(nx.path_graph(4))
    2
    >>> pared_index(nx.path_graph(4), return_cp=True)[0]
    2
    """
    pi = 0
    g1 = graph.copy()
    while True:
        n = g1.order()
        g1 = pared_graph(g1)
        if n != g1.order():
            pi += 1
        else:
            if return_cp:
                return (pi, g1)
            else:
                return pi


def completely_pared_graph(graph: nx.Graph) -> nx.Graph:
    """Successively remove dominated vertices until reaching a fixed point.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import completely_pared_graph
    >>> sorted(completely_pared_graph(nx.path_graph(4)).nodes())
    [3]
    """
    g1 = graph.copy()
    while True:
        n = g1.order()
        g1 = remove_dominated_vertex(g1)
        if n == g1.order():
            return g1


def is_dismantlable(graph: nx.Graph) -> bool:
    """Return True when the graph dismantles down to a single vertex.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.dominated import is_dismantlable
    >>> is_dismantlable(nx.path_graph(4))
    True
    >>> is_dismantlable(nx.cycle_graph(4))
    False
    """
    return completely_pared_graph(graph).order() == 1
