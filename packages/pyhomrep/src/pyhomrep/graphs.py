"""Graph constructions retained for compatibility with pyhomrep callers."""

import networkx as nx
from pycliques.named import matching_graph as _matching_graph


def matching_graph(n: int) -> nx.Graph:
    """Graph of disjoint edges of K_n; delegated to pycliques.named."""
    return _matching_graph(n)
