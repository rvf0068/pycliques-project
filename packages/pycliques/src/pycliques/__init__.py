"""Public package exports for the pycliques project."""

__version__ = "0.1.0"

from .cliques import Clique, clique_graph, homotopy_clique_graph
from .coaffinations import CoaffinePair, automorphisms, coaffinations
from .cutpoints import (
    cutpoint_edge_contractions,
    cutpoint_edge_removals,
    cutpoint_reductions,
    has_local_cutpoints,
    local_cutpoints,
    neighborhood_components,
    reduction_retracts_to,
)
from .dominated import (
    closed_neighborhood,
    completely_pared_graph,
    dominates,
    find_dominated_vertex,
    is_dismantlable,
    is_dominated_vertex,
    pared_graph,
    pared_index,
    remove_dominated_vertex,
    twin_classes,
)
from .helly import (
    is_clique_helly,
    is_hereditary_clique_helly,
    n_closed,
    n_open,
    u_closed,
    u_open,
)
from .named import (
    complement_of_cycle,
    graph_suspension,
    octahedron,
    snub_dysphenoid,
    suspension_of_cycle,
)
from .retractions import (
    dict_to_tuple,
    graph_from_gap_adjacency_list,
    has_induced,
    invert_dict,
    is_map,
    retraction,
    retracts,
    retracts_to,
    special_octahedra,
)
from .small import (
    CliqueSequence,
    Verdict,
    eventually_retracts_specially,
    is_eventually_helly,
)
from .surfaces import (
    is_closed_surface,
    is_cycle,
    is_path,
    is_regular,
    is_surface,
    open_neighborhood,
)

__all__ = [
    "__version__",
    # cliques
    "Clique",
    "clique_graph",
    "homotopy_clique_graph",
    # coaffinations
    "CoaffinePair",
    "automorphisms",
    "coaffinations",
    # cutpoints
    "cutpoint_edge_contractions",
    "cutpoint_edge_removals",
    "cutpoint_reductions",
    "has_local_cutpoints",
    "local_cutpoints",
    "neighborhood_components",
    "reduction_retracts_to",
    # dominated
    "closed_neighborhood",
    "completely_pared_graph",
    "dominates",
    "find_dominated_vertex",
    "is_dismantlable",
    "is_dominated_vertex",
    "pared_graph",
    "pared_index",
    "remove_dominated_vertex",
    "twin_classes",
    # helly
    "is_clique_helly",
    "is_hereditary_clique_helly",
    "n_closed",
    "n_open",
    "u_closed",
    "u_open",
    # named
    "complement_of_cycle",
    "graph_suspension",
    "octahedron",
    "snub_dysphenoid",
    "suspension_of_cycle",
    # retractions
    "dict_to_tuple",
    "graph_from_gap_adjacency_list",
    "has_induced",
    "invert_dict",
    "is_map",
    "retraction",
    "retracts",
    "retracts_to",
    "special_octahedra",
    # small
    "CliqueSequence",
    "Verdict",
    "eventually_retracts_specially",
    "is_eventually_helly",
    # surfaces
    "is_closed_surface",
    "is_cycle",
    "is_path",
    "is_regular",
    "is_surface",
    "open_neighborhood",
]
