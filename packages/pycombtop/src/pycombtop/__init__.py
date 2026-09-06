"""
pycombtop: Combinatorial topology and simplicial complexes.
"""

# Use relative imports to pull from your complex.py module
# fundamental_group
from .fundamental_group import (
    FundamentalRecord,
    covering_graph,
    first_homology_rank,
    fundamental_group,
    fundamental_group_as_permutation_group,
)
from .hom_complex import graph_homomorphisms, hom_graph

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
    is_contractible_via_flag_apex,
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
    antistar,
    bounded_degree,
    bounded_degree_complex,
    clique_complex,
    complex_of_forests,
    directed_neighborhood_complex,
    dong_matching,
    is_oriented_simplex,
    link,
    neighborhood_complex,
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
    "dong_matching",
    "nerve_of_sets",
    "clique_complex",
    "nerve_of_cliques",
    "bounded_degree",
    "bounded_degree_complex",
    "is_oriented_simplex",
    "oriented_complex",
    "complex_of_forests",
    "neighborhood_complex",
    "directed_neighborhood_complex",
    "link",
    "antistar",
    # fundamental_group
    "FundamentalRecord",
    "covering_graph",
    "first_homology_rank",
    "fundamental_group",
    "fundamental_group_as_permutation_group",
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
    "is_contractible_via_flag_apex",
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
    # hom_complex
    "graph_homomorphisms",
    "hom_graph",
]
