"""
pycombtop: Combinatorial topology and simplicial complexes.
"""

# Use relative imports to pull from your complex.py module
# fundamental_group
from .covering_divergence import (
    TriangularCoverDivergenceCertificate,
    verify_triangular_cover_divergence_certificate,
)
from .covering_divergence_search import (
    CoveringSearchResult,
    CoveringSearchStatus,
    search_covering_divergence,
)
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
    homotopy_type_clique_graph_via_independent_neighborhood,
    homotopy_type_large_graph,
    homotopy_type_sc_with_verdict,
    homotopy_type_with_verdict,
    intersection_complex,
    is_contractible_via_flag_apex,
    is_vertex_decomposable,
    star,
    star_cluster,
)
from .integral_morse import (
    IntegralMorseCertificate,
    bipartite_shore_facets,
    integral_morse_certificate,
    verify_integral_morse,
)
from .morse_certificates import (
    MorseCertificate,
    facet_morse_certificate,
    graph_morse_certificate,
)
from .recursive_wedges import (
    RecursiveWedgeCertificate,
    recursive_wedge_certificate,
    verify_recursive_wedge,
)
from .relative_collapse import (
    CollapseResult,
    CollapseStatus,
    relative_collapse,
    verify_relative_collapse,
)
from .s_collapses import (
    complete_s_collapse,
    complete_s_collapse_edges,
    complete_s_collapse_edges_with_certificate,
    complete_s_collapse_with_certificate,
    dismantling_certificate,
    has_s_dismantlable_edge,
    has_s_dismantlable_vertex,
    is_s_dismantlable_edge,
    is_s_dismantlable_vertex,
    remove_s_dismantlable_edge,
    remove_s_dismantlable_vertex,
    verify_dismantling_certificate,
    verify_s_collapse_move,
)
from .simplex import (
    Simplex,
    SimplicialComplex,
    all_subsets,
    bounded_degree,
    bounded_degree_complex,
    clique_complex,
    complex_of_forests,
    deletion,
    directed_neighborhood_complex,
    dong_matching,
    is_oriented_simplex,
    link,
    neighborhood_complex,
    nerve_of_cliques,
    nerve_of_sets,
    oriented_complex,
    relative_betti_numbers,
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
    "deletion",
    "relative_betti_numbers",
    # fundamental_group
    "FundamentalRecord",
    "covering_graph",
    "first_homology_rank",
    "fundamental_group",
    "fundamental_group_as_permutation_group",
    # covering divergence
    "TriangularCoverDivergenceCertificate",
    "verify_triangular_cover_divergence_certificate",
    "CoveringSearchResult",
    "CoveringSearchStatus",
    "search_covering_divergence",
    # homotopy_type
    "MorseCertificate",
    "facet_morse_certificate",
    "graph_morse_certificate",
    "IntegralMorseCertificate",
    "integral_morse_certificate",
    "verify_integral_morse",
    "bipartite_shore_facets",
    "RecursiveWedgeCertificate",
    "recursive_wedge_certificate",
    "verify_recursive_wedge",
    "HomotopyVerdict",
    "Theorem",
    "betti_numbers_graph",
    "betti_numbers_sc",
    "collapse",
    "homotopy_type_clique_graph_via_independent_neighborhood",
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
    "complete_s_collapse_with_certificate",
    "complete_s_collapse_edges",
    "complete_s_collapse_edges_with_certificate",
    "dismantling_certificate",
    "has_s_dismantlable_edge",
    "has_s_dismantlable_vertex",
    "is_s_dismantlable_edge",
    "is_s_dismantlable_vertex",
    "remove_s_dismantlable_edge",
    "remove_s_dismantlable_vertex",
    "verify_dismantling_certificate",
    "verify_s_collapse_move",
    # hom_complex
    "graph_homomorphisms",
    "hom_graph",
    "CollapseResult",
    "CollapseStatus",
    "relative_collapse",
    "verify_relative_collapse",
]
