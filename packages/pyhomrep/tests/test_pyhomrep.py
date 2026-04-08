"""Tests for the pyhomrep package."""

import networkx as nx
import pyhomrep
from pycliques import clique_graph
from pycombtop import clique_complex
from pyhomrep.decomposition import MTS, blockI
from pyhomrep.graphs import matching_graph
from pyhomrep.matrix import regular_representation
from pyhomrep.pchains import PChain
from pyhomrep.simplicial import faces, p_simplex
from pyhomrep.utilities import (
    boundary_op_n,
    form_matrix_yt,
    make_permutation,
    partitions_list,
    reduce_matrix,
    size_conjugacy_class,
    tuple_sorted,
)
from pyhomrep.young import YoungTableaux
from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup


def test_import():
    """Test that the package imports correctly."""
    assert pyhomrep


class TestPChain:
    """Tests for the PChain class."""

    def test_creation(self):
        """Test creating p-chains."""
        p = PChain([(0, 1, 2), (0, 1, 3)], [-1, 2])
        assert p.dic == {(0, 1, 2): -1, (0, 1, 3): 2}

    def test_addition(self):
        """Test adding p-chains."""
        p1 = PChain([(0, 1, 2)], [2])
        p2 = PChain([(0, 1, 3)], [5])
        result = p1 + p2
        assert result.dic == {(0, 1, 2): 2, (0, 1, 3): 5}

    def test_subtraction(self):
        """Test subtracting p-chains."""
        p1 = PChain([(3, 4, 5)], [3])
        p2 = PChain([(1, 8, 9)], [1])
        result = p1 - p2
        assert result.dic == {(3, 4, 5): 3, (1, 8, 9): -1}

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        p = PChain([(7, 8, 9), (10, 11, 12)], [3, 2])
        result = 3 * p
        assert result.dic == {(7, 8, 9): 9, (10, 11, 12): 6}


class TestYoungTableaux:
    """Tests for the YoungTableaux class."""

    def test_MNR(self):
        """Test generating border-strip tableaux."""
        yt = YoungTableaux([4, 1], [3, 2])
        assert yt.MNR() == [[[1, 1, 2, 2], [1]]]

    def test_heights(self):
        """Test computing heights."""
        yt = YoungTableaux([5, 2, 1], [3, 3, 1, 1])
        assert yt.heights() == [1, 1, 1, 2, 2, 3]

    def test_CMNR(self):
        """Test Murnaghan-Nakayama rule computation."""
        yt = YoungTableaux([5, 2, 1], [3, 3, 1, 1])
        assert yt.CMNR() == -2


class TestUtilities:
    """Tests for utility functions."""

    def test_tuple_sorted(self):
        """Test recursive tuple sorting."""
        a1 = ((6, 5), (1, 0), (3, 2))
        assert tuple_sorted(a1) == ((0, 1), (2, 3), (5, 6))

    def test_partitions_list(self):
        """Test partition generation."""
        assert partitions_list(3) == [[3], [1, 1, 1], [2, 1]]

    def test_make_permutation(self):
        """Test creating permutations from partitions."""
        p = make_permutation([5])
        assert p == Permutation(0, 1, 2, 3, 4)

    def test_size_conjugacy_class(self):
        """Test computing conjugacy class sizes."""
        assert size_conjugacy_class([4], 4) == 6
        assert size_conjugacy_class([1, 1, 1, 1], 4) == 1
        assert size_conjugacy_class([2, 2], 4) == 3

    def test_form_matrix_yt(self):
        """Test forming character table matrix."""
        v = partitions_list(3)
        M = form_matrix_yt(v)
        assert M[0, 0] == 1
        assert M[2, 1] == 2


class TestMatrixRepresentation:
    """Tests for matrix representations."""

    def test_regular_representation(self):
        """Test creating regular representation."""
        G = SymmetricGroup(3)
        rr = regular_representation(G)
        assert rr.degree == 6

    def test_is_unitary(self):
        """Test checking if representation is unitary."""
        G = SymmetricGroup(3)
        rr = regular_representation(G)
        assert rr.is_unitary() is True


class TestDecomposition:
    """Tests for decomposition functions."""

    def test_MTS(self):
        """Test MTS decomposition."""
        from sympy.matrices import Matrix

        M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
        A = M.H * M
        V = MTS(A)
        result = V.H * A * V
        # Check it's the identity
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert result[i, j] == 1
                else:
                    assert result[i, j] == 0

    def test_blockI(self):
        """Test blockI function."""
        from sympy.matrices import Matrix

        M = Matrix([[1, 1, 1], [1, 1, 1]])
        result = blockI(M, 4, 0)
        assert result[0, 0] == 1
        assert result[3, 3] == 1

    def test_reduce_matrix(self):
        """Test reduce_matrix function."""
        from sympy.matrices import Matrix

        M = Matrix(
            [
                [-1, -1, -1, -1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0],
                [0, 1, 0, 0, 1, 0, -1, -1],
                [0, 0, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
            ]
        )
        P, R = reduce_matrix(M)
        assert P * M == M.rref()[0]


class TestGraphs:
    """Tests for graph functions."""

    def test_matching_graph(self):
        """Test matching graph construction."""
        G = matching_graph(4)
        assert G.number_of_nodes() == 6
        assert G.number_of_edges() == 3

    def test_clique_graph(self):
        """Test clique graph construction."""
        G = nx.cycle_graph(4)
        K = clique_graph(G)
        assert K.number_of_nodes() == 4


class TestSimplicialComplex:
    """Tests for simplicial complex functions."""

    def test_faces(self):
        """Test computing faces."""
        G = nx.complete_graph(3)
        sc = clique_complex(G)
        f = faces(sc)
        assert len(f) == 8  # empty + 3 vertices + 3 edges + 1 triangle

    def test_dimension(self):
        """Test computing dimension."""
        G = nx.complete_graph(4)
        sc = clique_complex(G)
        assert sc.dimension() == 3

    def test_p_simplex(self):
        """Test getting p-simplices."""
        G = nx.complete_graph(3)
        sc = clique_complex(G)
        assert len(p_simplex(sc, 0)) == 3
        assert len(p_simplex(sc, 1)) == 3
        assert len(p_simplex(sc, 2)) == 1


class TestBoundary:
    """Tests for boundary operator."""

    def test_boundary_operator_vanishes(self):
        """Test that boundary^2 = 0."""
        v = PChain([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], [1, 1, 1, 1])
        u = boundary_op_n(v)
        result = boundary_op_n(u)
        # All coefficients should be 0
        for val in result.dic.values():
            assert val == 0

    def test_boundary_of_0_chain(self):
        """Test that boundary of 0-chain is empty."""
        w = PChain([(0,), (1,), (2,)], [1, 1, 1])
        result = boundary_op_n(w)
        assert result.dic == {}


class TestDecomposeIntoIrreducibles:
    """Tests for decomposition of homology into irreducible representations."""

    def test_decompose_function_exists(self):
        """Test that decompose_into_irreducibles is importable."""
        from pyhomrep.simplicial import decompose_into_irreducibles

        assert callable(decompose_into_irreducibles)

    def test_matching_graph_structure(self):
        """Test matching graph properties used for decomposition.

        The matching graph M(n) has C(n,2) = n(n-1)/2 vertices representing
        edges of K_n, connected when the corresponding edges are disjoint.
        """
        # M(4) is the Petersen graph's complement structure
        G4 = matching_graph(4)
        assert G4.number_of_nodes() == 6  # C(4,2) = 6 edges in K_4
        assert G4.number_of_edges() == 3  # 3 perfect matchings of K_4

        # The clique complex captures the structure
        sc4 = clique_complex(G4)
        # Should have some 0-simplices (vertices) and possibly higher
        assert sc4.dimension() >= 0

    def test_character_p_homology_basic(self):
        """Test character computation for trivial cases."""
        from pyhomrep.simplicial import (
            basis_group_oriented_p_chains,
        )

        # Simple triangle
        G = nx.complete_graph(3)
        sc = clique_complex(G)

        # Check basis functions work
        basis_0 = basis_group_oriented_p_chains(sc, 0)
        assert basis_0 is not None
        assert len(basis_0.dic) == 3  # 3 vertices

        basis_1 = basis_group_oriented_p_chains(sc, 1)
        assert basis_1 is not None
        assert len(basis_1.dic) == 3  # 3 edges
