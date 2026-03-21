"""
pycombtop: Combinatorial topology and simplicial complexes.
"""

# Use relative imports to pull from your complex.py module
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
]
