"""
pycombtop: Combinatorial topology and simplicial complexes.
"""

# Use relative imports to pull from your complex.py module
# homotopy_type
from .homotopy_type import (
    HomotopyVerdict,
    Theorem,
    WedgeOfSpheres,
    betti_numbers_graph,
    betti_numbers_sc,
    collapse,
    homotopy_type_large_graph,
    homotopy_type_sc_with_verdict,
    homotopy_type_with_verdict,
    intersection_complex,
    is_vertex_decomposable,
    star,
    star_cluster,
)
from .s_collapses import (
    complete_s_collapse,
    complete_s_collapse_edges,
    has_s_dismantlable_edge,
    has_s_dismantlable_vertex,
    is_s_dismantlable_edge,
    is_s_dismantlable_vertex,
    remove_s_dismantlable_edge,
    remove_s_dismantlable_vertex,
)
from .simplex import (
    Simplex,
    SimplicialComplex,
    all_subsets,
    bounded_degree,
    bounded_degree_complex,
    clique_complex,
    complex_of_forests,
    is_oriented_simplex,
    nerve_of_cliques,
    nerve_of_sets,
    oriented_complex,
)

# Explicitly declare the public API.
# This controls what happens if someone runs `from pycombtop import *`
__all__ = [
    # simplex
    "Simplex",
    "SimplicialComplex",
    "all_subsets",
    "nerve_of_sets",
    "clique_complex",
    "nerve_of_cliques",
    "bounded_degree",
    "bounded_degree_complex",
    "is_oriented_simplex",
    "oriented_complex",
    "complex_of_forests",
    # homotopy_type
    "HomotopyVerdict",
    "Theorem",
    "betti_numbers_graph",
    "betti_numbers_sc",
    "collapse",
    "homotopy_type_large_graph",
    "homotopy_type_sc_with_verdict",
    "homotopy_type_with_verdict",
    "intersection_complex",
    "is_vertex_decomposable",
    "star",
    "star_cluster",
    "WedgeOfSpheres",
    # s_collapses
    "complete_s_collapse",
    "complete_s_collapse_edges",
    "has_s_dismantlable_edge",
    "has_s_dismantlable_vertex",
    "is_s_dismantlable_edge",
    "is_s_dismantlable_vertex",
    "remove_s_dismantlable_edge",
    "remove_s_dismantlable_vertex",
]
