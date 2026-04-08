"""pyhomrep: Group representations on simplicial homology.

A Python package based on sympy for representation theory on homologies of
simplicial complexes.
"""

# P-chains
# Decomposition
from pyhomrep.decomposition import (
    MTS,
    block,
    blockI,
    is_irreducible,
    reduce,
    unitary_representation,
)

# Graph utilities
from pyhomrep.graphs import matching_graph

# Matrix representations
from pyhomrep.matrix import MatrixRepresentation, regular_representation
from pyhomrep.pchains import PChain

# Simplicial complex functions for representation theory
from pyhomrep.simplicial import (
    basis_group_oriented_p_chains,
    character_image,
    character_kernel,
    character_matrix_permutation,
    character_p_homology,
    decompose_into_irreducibles,
    faces,
    image_boundary_op,
    kernel_boundary_op,
    matrix_symmetric_representate,
    p_simplex,
    representate_in_simplex,
)

# Utilities
from pyhomrep.utilities import (
    boundary_op_n,
    columnspace,
    form_matrix_yt,
    have_same_elements,
    make_permutation,
    nullspace,
    orientation_function,
    partitions_list,
    permutation_in_simplex_test,
    reduce_matrix,
    size_conjugacy_class,
    tuple_permutation,
    tuple_sorted,
)

# Young tableaux and Murnaghan-Nakayama rule
from pyhomrep.young import YoungTableaux

__all__ = [
    # Core classes
    "PChain",
    "YoungTableaux",
    "MatrixRepresentation",
    # Matrix representation functions
    "regular_representation",
    # Decomposition functions
    "MTS",
    "block",
    "blockI",
    "is_irreducible",
    "reduce",
    "unitary_representation",
    # Graph functions
    "matching_graph",
    # Simplicial complex functions
    "faces",
    "p_simplex",
    "basis_group_oriented_p_chains",
    "representate_in_simplex",
    "matrix_symmetric_representate",
    "kernel_boundary_op",
    "image_boundary_op",
    "character_kernel",
    "character_image",
    "character_p_homology",
    "decompose_into_irreducibles",
    "character_matrix_permutation",
    # Utilities
    "reduce_matrix",
    "boundary_op_n",
    "columnspace",
    "have_same_elements",
    "form_matrix_yt",
    "make_permutation",
    "nullspace",
    "orientation_function",
    "partitions_list",
    "permutation_in_simplex_test",
    "size_conjugacy_class",
    "tuple_permutation",
    "tuple_sorted",
]
