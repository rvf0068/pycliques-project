"""Public package exports for the pycliques project."""

__version__ = "0.1.0"

from .cliques import Clique, clique_graph, homotopy_clique_graph
from .coaffinations import CoaffinePair, automorphisms, coaffinations

__all__ = [
    "__version__",
    "Clique",
    "clique_graph",
    "homotopy_clique_graph",
    "CoaffinePair",
    "automorphisms",
    "coaffinations",
]
