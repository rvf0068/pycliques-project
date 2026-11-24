"""
pycombtop: Combinatorial topology and simplicial complexes.
"""

# Use relative imports to pull from your complex.py module
# fundamental_group
from .closed_trace_contractibility import (
    ClosedTraceResult,
    check_closed_trace_contractibility,
)
from .covering_divergence import (
    TriangularCoverDivergenceCertificate,
    verify_triangular_cover_divergence_certificate,
)
from .covering_divergence_search import (
    CoveringSearchResult,
    CoveringSearchStatus,
    search_covering_divergence,
)
from .facet_punctures import (
    FacetPunctureWedgeCertificate,
    facet_puncture_wedge_certificate,
    verify_facet_puncture_wedge,
)
from .flag_edge_star import (
    FlagMove,
    FlagReductionCertificate,
    ReducedStarWedgeCertificate,
    VertexStarWedgeCertificate,
    reduce_flag_graph,
    reduced_star_wedge_certificate,
    verify_flag_reduction,
    verify_reduced_star_wedge,
    verify_vertex_star_wedge,
    vertex_star_wedge_certificate,
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
from .hopf_wedge import (
    HopfWedgeCertificate,
    ReducedHopfWedgeCertificate,
    hopf_wedge_certificate,
    reduced_hopf_wedge_certificate,
    verify_hopf_wedge,
    verify_reduced_hopf_wedge,
)
from .indexed_relative_collapse import (
    indexed_relative_collapse,
    verify_indexed_relative_collapse,
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
from .strong_collapses import (
    UniformStrongCollapseCertificate,
    uniform_strong_collapse_certificate,
    verify_uniform_strong_collapse_certificate,
)
from .symbolic_dong import SymbolicDongCertificate, symbolic_dong_certificate
from .symbolic_integral_morse import (
    SymbolicIntegralMorseCertificate,
    symbolic_integral_morse_certificate,
)
from .weighted_conicity import (
    WeightedConicityCertificate,
    find_weighted_conicity_certificate,
    verify_weighted_conicity,
)

# Explicitly declare the public API.
# This controls what happens if someone runs `from pycombtop import *`
__all__ = [
    "FacetPunctureWedgeCertificate",
    "facet_puncture_wedge_certificate",
    "verify_facet_puncture_wedge",
    # certified flag-edge reductions and vertex-star homotopy
    "FlagMove",
    "FlagReductionCertificate",
    "VertexStarWedgeCertificate",
    "ReducedStarWedgeCertificate",
    "reduce_flag_graph",
    "verify_flag_reduction",
    "vertex_star_wedge_certificate",
    "verify_vertex_star_wedge",
    "reduced_star_wedge_certificate",
    "verify_reduced_star_wedge",
    # Hopf attaching-map and cup-square certificates
    "HopfWedgeCertificate",
    "ReducedHopfWedgeCertificate",
    "hopf_wedge_certificate",
    "verify_hopf_wedge",
    "reduced_hopf_wedge_certificate",
    "verify_reduced_hopf_wedge",
    # simplex
    "Simplex",
    "SimplicialComplex",
    "all_subsets",
    "dong_matching",
    "SymbolicDongCertificate",
    "symbolic_dong_certificate",
    "SymbolicIntegralMorseCertificate",
    "symbolic_integral_morse_certificate",
    "WeightedConicityCertificate",
    "find_weighted_conicity_certificate",
    "verify_weighted_conicity",
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
    "ClosedTraceResult",
    "check_closed_trace_contractibility",
    "UniformStrongCollapseCertificate",
    "uniform_strong_collapse_certificate",
    "verify_uniform_strong_collapse_certificate",
    "indexed_relative_collapse",
    "verify_indexed_relative_collapse",
    "verify_relative_collapse",
]
