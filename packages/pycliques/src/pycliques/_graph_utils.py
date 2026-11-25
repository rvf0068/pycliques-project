"""Common input checks for undirected graph algorithms."""

import networkx as nx


def is_simple_graph(graph: object) -> bool:
    return (
        isinstance(graph, nx.Graph)
        and not graph.is_directed()
        and not graph.is_multigraph()
        and not nx.number_of_selfloops(graph)
    )


def same_labeled_graph(left: nx.Graph, right: nx.Graph) -> bool:
    """Compare undirected adjacency on the given labels, ignoring attributes."""
    return set(left) == set(right) and {frozenset(e) for e in left.edges()} == {
        frozenset(e) for e in right.edges()
    }
