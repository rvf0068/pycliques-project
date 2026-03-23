"""Generators for specific graph families."""

import networkx as nx
from networkx.algorithms.operators.unary import complement


def graph_suspension(graph: nx.Graph) -> nx.Graph:
    """Return the suspension of ``graph``.

    The suspension is obtained by adjoining two new vertices (labelled 0
    and 1) that are made adjacent to every vertex of ``graph``.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Returns

    networkx.Graph
        The suspension graph.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.named import graph_suspension
    >>> sorted(graph_suspension(nx.empty_graph(3)).edges())
    [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
    """
    mapping = {v: v + 2 for v in graph.nodes()}
    result = nx.Graph()
    result.add_nodes_from([0, 1])
    for v in graph.nodes():
        result.add_node(mapping[v])
    for u, v in graph.edges():
        result.add_edge(mapping[u], mapping[v])
    for v in graph.nodes():
        result.add_edge(0, mapping[v])
        result.add_edge(1, mapping[v])
    return result


def suspension_of_cycle(n: int) -> nx.Graph:
    """Return the suspension of the cycle graph :math:`C_n`.

    .. rubric:: Parameters

    n : int
        Number of vertices in the cycle.

    .. rubric:: Returns

    networkx.Graph
        The suspension of :math:`C_n`.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.named import suspension_of_cycle
    >>> nx.is_isomorphic(nx.octahedral_graph(), suspension_of_cycle(4))
    True
    """
    return graph_suspension(nx.cycle_graph(n))


def complement_of_cycle(n: int) -> nx.Graph:
    """Return the complement of the cycle graph :math:`C_n`.

    .. rubric:: Parameters

    n : int
        Number of vertices in the cycle.

    .. rubric:: Returns

    networkx.Graph
        The complement of :math:`C_n`.

    .. rubric:: Examples

    >>> from pycliques.named import complement_of_cycle
    >>> complement_of_cycle(5).number_of_nodes()
    5
    >>> complement_of_cycle(5).number_of_edges()
    5
    """
    return complement(nx.cycle_graph(n))


def octahedron(n: int) -> nx.Graph:
    """Return the *n*-th octahedron (complement of *n* disjoint edges).

    .. rubric:: Parameters

    n : int
        Number of disjoint edges whose complement is taken.

    .. rubric:: Returns

    networkx.Graph
        The complement of :math:`nK_2`.

    .. rubric:: Examples

    >>> from pycliques.named import octahedron
    >>> nx.is_isomorphic(nx.octahedral_graph(), octahedron(3))
    True
    >>> sorted(nx.complement(octahedron(4)).edges())
    [(0, 1), (2, 3), (4, 5), (6, 7)]
    """
    edges = [nx.complete_graph(2) for _ in range(n)]
    return nx.complement(nx.disjoint_union_all(edges))


def snub_dysphenoid() -> nx.Graph:
    """Return the snub dysphenoid graph.

    .. rubric:: Returns

    networkx.Graph
        The snub dysphenoid on 8 vertices.

    .. rubric:: Examples

    >>> from pycliques.named import snub_dysphenoid
    >>> snub_dysphenoid().number_of_nodes()
    8
    """
    return nx.from_graph6_bytes(bytes("GQyuzw", "utf8"))
