"""Utility functions for homology representation computations.

This module provides various utility functions for manipulating tuples,
permutations, simplices, and computing boundary operators.

Mathematical Background
-----------------------
The boundary operator ∂_p: C_p(K) → C_{p-1}(K) is defined on oriented
simplices as:

    ∂_p[v_0, ..., v_p] = Σ_{i=0}^{p} (-1)^i [v_0, ..., v̂_i, ..., v_p]

where v̂_i denotes the omission of vertex v_i.

References
----------
- Hatcher, A. "Algebraic Topology", Chapter 2
- Munkres, J. "Elements of Algebraic Topology"
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from sympy import eye
from sympy.combinatorics import Permutation
from sympy.combinatorics.partitions import IntegerPartition
from sympy.core import S
from sympy.matrices import Matrix, zeros

from pyhomrep.pchains import PChain
from pyhomrep.young import YoungTableaux

if TYPE_CHECKING:
    from sympy.matrices.dense import MutableDenseMatrix


def tuple_sorted(a: Any) -> Any:
    """Sort tuples recursively, including nested tuples.

    The standard ``sorted`` function does not sort tuples of tuples correctly.
    This function handles nested structures.

    .. rubric:: Parameters

    a : tuple or int
        A tuple (possibly nested) or an integer.

    .. rubric:: Returns

    tuple or int
        The sorted tuple or the unchanged integer.

    .. rubric:: Examples

    >>> from pyhomrep.utilities import tuple_sorted
    >>> a1 = ((6, 5), (1, 0), (3, 2))
    >>> tuple_sorted(a1)
    ((0, 1), (2, 3), (5, 6))
    >>> a2 = (((4, 2), (1, 0), (5, 3)), ((2, 3), (1, 0), (6, 4)))
    >>> tuple_sorted(a2)
    (((0, 1), (2, 3), (4, 6)), ((0, 1), (2, 4), (3, 5)))
    """
    if isinstance(a, int):
        return a
    if isinstance(a[0], int):
        return tuple(sorted(a))
    else:
        w: list = []
        for b in a:
            w.append(tuple(tuple_sorted(b)))
        return tuple(sorted(tuple(w)))


def tuple_permutation(v: tuple, p: Permutation) -> tuple:
    """Apply a permutation to a tuple (possibly nested).

    .. rubric:: Parameters

    v : tuple
        The tuple to permute.
    p : Permutation
        The sympy Permutation to apply.

    .. rubric:: Returns

    tuple
        The permuted tuple.

    .. rubric:: Examples

    >>> from pyhomrep.utilities import tuple_permutation
    >>> from sympy.combinatorics import Permutation
    >>> a1 = (0, 1, 2, 3, 4)
    >>> tuple_permutation(a1, Permutation(0, 1, 2))
    (1, 2, 0, 3, 4)
    >>> a2 = ((2, 4), (1, 5), (3, 0))
    >>> tuple_permutation(a2, Permutation(1, 3))
    ((2, 4), (3, 5), (1, 0))
    """
    u: list = []
    w = list(v)
    test = True
    for i in range(len(v)):
        if isinstance(v[i], int):
            if v[i] in p:
                w[i] = p(v[i])
        else:
            u.append(tuple_permutation(tuple(v[i]), p))
            test = False
    if test:
        return tuple(w)
    else:
        return tuple(u)


def have_same_elements(a: Any, b: Any) -> bool:
    """Check if two tuples contain the same elements (orientation-equivalent).

    Two simplices are considered orientation-equivalent if they contain the
    same vertices, regardless of their ordering. This is used to determine
    if two representations of a simplex differ only by orientation.

    .. rubric:: Parameters

    a : tuple or int
        The first tuple or integer.
    b : tuple or int
        The second tuple or integer.

    .. rubric:: Returns

    bool
        True if the tuples contain the same elements.

    .. rubric:: Examples

    >>> from pyhomrep.utilities import have_same_elements
    >>> a1 = ((0, 1), (2, 3), (5, 6))
    >>> b1 = ((0, 3), (2, 1), (5, 6))
    >>> have_same_elements(a1, b1)
    False
    >>> a2 = ((0, 1), (2, 3), (5, 6))
    >>> b2 = ((6, 5), (1, 0), (3, 2))
    >>> have_same_elements(a2, b2)
    True
    """
    if isinstance(a, int):
        return bool(a == b)
    if isinstance(a[0], int):
        return bool(set(a) == set(b))
    else:
        for i in range(len(a)):
            if not any(have_same_elements(a[i], b[j]) for j in range(len(b))):
                return False
        return True


def orientation_function(a: tuple, b: tuple, p: int) -> bool:
    """Determine the orientation of b relative to a.

    .. rubric:: Parameters

    a : tuple
        The first tuple (reference orientation).
    b : tuple
        The second tuple to compare.
    p : int
        The dimension of the simplex.

    .. rubric:: Returns

    bool
        True if b has the same orientation as a, False otherwise.

    .. rubric:: Examples

    >>> from pyhomrep.utilities import orientation_function
    >>> a = ((0, 1), (2, 3), (5, 6))
    >>> b = ((6, 5), (1, 0), (3, 2))
    >>> orientation_function(a, b, 2)
    True
    """
    if p == 0:
        return True
    else:
        v = np.zeros((len(a),), dtype=int)
        for i in range(len(a)):
            for j in range(len(b)):
                if have_same_elements(a[i], b[j]):
                    v[j] = i
        perm = Permutation(list(v))
        return bool(perm.is_even)


def boundary_op_n(v: PChain) -> PChain:
    """Apply the boundary operator to a p-chain.

    .. rubric:: Parameters

    v : PChain
        A p-chain to apply the boundary operator to.

    .. rubric:: Returns

    PChain
        The resulting (p-1)-chain under the boundary operator.

    .. rubric:: Examples

    >>> from pyhomrep.pchains import PChain
    >>> from pyhomrep.utilities import boundary_op_n
    >>> w = PChain([(0,), (1,), (2,), (3,)], [1, 1, 1, 1])
    >>> boundary_op_n(w).dic
    {}
    >>> v = PChain([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], [1, 1, 1, 1])
    >>> u = boundary_op_n(v)
    >>> boundary_op_n(u).dic  # boundary^2 = 0
    {}
    """
    if not v.dic:
        return PChain([], [])
    p = len(list(v.dic.keys())[0]) - 1
    s = PChain([], [])
    if p != 0:
        for u in v.dic.keys():
            c = 0
            for i in u:
                w = list(u)
                w.remove(i)
                if orientation_function(tuple(tuple_sorted(tuple(w))), tuple(w), p):
                    s1 = PChain([tuple(tuple_sorted(tuple(w)))], [abs(v.dic[u])])
                    if np.sign(v.dic[u] * (-1) ** c) < 0:
                        s = s - s1
                    else:
                        s = s + s1
                else:
                    s1 = PChain([tuple(tuple_sorted(tuple(w)))], [abs(v.dic[u])])
                    if np.sign(v.dic[u] * (-1) ** (c + 1)) < 0:
                        s = s - s1
                    else:
                        s = s + s1
                c = c + 1
        return s
    else:
        return s


def nullspace(m: Matrix) -> list[list[Any]]:
    """Compute the nullspace (kernel) of a matrix.

    The kernel (nullspace) of a matrix M is the set of vectors v such that
    Mv = 0. This computes a basis for that subspace.

    .. rubric:: Parameters

    m : Matrix
        The sympy matrix.

    .. rubric:: Returns

    list
        A list of vectors spanning the nullspace.

    .. rubric:: Examples

    >>> from sympy.matrices import Matrix
    >>> from pyhomrep.utilities import nullspace
    >>> M1 = Matrix([[2, 4, 6, 6], [8, 20, 0, 1], [5, 0, 3, 2]])
    >>> nullspace(M1)
    [[3/16, -1/8, -47/48, 1]]
    """
    u = m.nullspace()
    w = [list(g) for g in u]
    if not w:
        return [list(np.zeros((m.shape[1],), dtype=int))]
    return w


def columnspace(m: Matrix) -> list[list[Any]]:
    """Compute the column space (image) of a matrix.

    The column space (image) of a matrix M is the span of its column vectors.
    This computes a basis for that subspace.

    .. rubric:: Parameters

    m : Matrix
        The sympy matrix.

    .. rubric:: Returns

    list
        A list of vectors spanning the column space.

    .. rubric:: Examples

    >>> from sympy.matrices import Matrix
    >>> from pyhomrep.utilities import columnspace
    >>> M1 = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> len(columnspace(M1))
    1
    """
    u = m.columnspace()
    w = [list(g) for g in u]
    if not w:
        return [list(np.zeros((m.shape[0],), dtype=int))]
    return w


def reduce_matrix(n: Matrix) -> tuple[MutableDenseMatrix, MutableDenseMatrix]:
    """Compute row reduced form and the transformation matrix.

    Uses Gauss-Jordan elimination to reduce a matrix to row echelon form
    while tracking the elementary row operations.

    .. rubric:: Parameters

    n : Matrix
        The matrix to reduce.

    .. rubric:: Returns

    tuple
        A tuple (P, R) where P is the product of elementary matrices and R is
        the row reduced form, such that P @ original = R.

    .. rubric:: Examples

    >>> from sympy.matrices import Matrix
    >>> from pyhomrep.utilities import reduce_matrix
    >>> M = Matrix([[-1, -1, -1, -1, 0, 0, 0, 0],
    ...             [1, 0, 0, 0, -1, -1, 0, 0],
    ...             [0, 1, 0, 0, 1, 0, -1, -1],
    ...             [0, 0, 1, 0, 0, 1, 1, 0],
    ...             [0, 0, 0, 1, 0, 0, 0, 1]])
    >>> P, R = reduce_matrix(M)
    >>> P * M == M.rref()[0]
    True
    """
    m = n.copy()
    lead = 0
    row_count = m.shape[0]
    column_count = m.shape[1]
    a = eye(row_count)

    for r in range(row_count):
        b1 = eye(row_count)
        if column_count <= lead:
            return a, m
        i = r
        while m[i, lead] == 0:
            i = i + 1
            if row_count == i:
                i = r
                lead = lead + 1
                if column_count == lead:
                    return a, m
        b1.row_swap(i, r)
        m.row_swap(i, r)
        x = m[r, lead]
        for k in range(column_count):
            m[r, k] = S(m[r, k]) / x
            if k < row_count:
                b1[r, k] = S(b1[r, k]) / x
        for i in range(row_count):
            if i != r:
                x = m[i, lead]
                for k in range(column_count):
                    m[i, k] = m[i, k] - m[r, k] * x
                    if k < row_count:
                        b1[i, k] = b1[i, k] - b1[r, k] * x
        lead = lead + 1
        a = b1 * a
    return a, m


def permutation_in_simplex_test(vec: PChain, p: Permutation) -> PChain:
    """Apply a permutation to a p-chain.

    .. rubric:: Parameters

    vec : PChain
        A p-chain to transform.
    p : Permutation
        The permutation to apply.

    .. rubric:: Returns

    PChain
        A new p-chain with the permutation applied to all simplices.

    .. rubric:: Examples

    >>> from pyhomrep.pchains import PChain
    >>> from pyhomrep.utilities import permutation_in_simplex_test
    >>> from sympy.combinatorics import Permutation
    >>> v = PChain([(0, 1, 2)], [1])
    >>> permutation_in_simplex_test(v, Permutation(0, 1)).dic
    {(0, 1, 2): -1}
    """
    s = PChain([], [])
    if vec.dic:
        v = list(vec.dic.keys())
        p_dim = len(list(vec.dic.keys())[0]) - 1
        for a in v:
            if isinstance(a, int):
                return vec
            else:
                w = tuple_permutation(a, p)
                if orientation_function(tuple_sorted(w), w, p_dim):
                    s = s + PChain([tuple(tuple_sorted(w))], [vec.dic[a]])
                else:
                    s = s - PChain([tuple(tuple_sorted(w))], [vec.dic[a]])
        return s
    else:
        return s


def partitions_list(n: int) -> list[list[int]]:
    """Generate all partitions of an integer.

    .. rubric:: Parameters

    n : int
        The integer to partition.

    .. rubric:: Returns

    list
        A list of partitions, where each partition is a list of integers.

    .. rubric:: Examples

    >>> from pyhomrep.utilities import partitions_list
    >>> partitions_list(3)
    [[3], [1, 1, 1], [2, 1]]
    >>> partitions_list(4)
    [[4], [1, 1, 1, 1], [2, 1, 1], [2, 2], [3, 1]]
    """
    p = IntegerPartition([n])
    w: list[list[int]] = []
    while list(p.args[1]) not in w:
        w.append(list(p.args[1]))
        p = p.next_lex()
    return w


def form_matrix_yt(w: list[list[int]]) -> Matrix:
    """Form the character table matrix for a symmetric group.

    .. rubric:: Parameters

    w : list
        A list of partitions.

    .. rubric:: Returns

    Matrix
        The character table matrix.

    .. rubric:: Examples

    >>> from pyhomrep.utilities import partitions_list, form_matrix_yt
    >>> v = partitions_list(3)
    >>> form_matrix_yt(v)
    Matrix([
    [ 1, 1,  1],
    [ 1, 1, -1],
    [-1, 2,  0]])
    """
    m = zeros(len(w), len(w))
    for i in range(len(w)):
        for j in range(len(w)):
            m[i, j] = YoungTableaux(w[i], w[j]).CMNR()
    return m


def make_permutation(partition: list[int]) -> Permutation:
    """Create a representative permutation for a conjugacy class.

    .. rubric:: Parameters

    partition : list
        A partition representing a conjugacy class.

    .. rubric:: Returns

    Permutation
        A representative permutation of the conjugacy class.

    .. rubric:: Examples

    >>> from pyhomrep.utilities import make_permutation
    >>> make_permutation([5])
    Permutation(0, 1, 2, 3, 4)
    >>> make_permutation([1, 1, 1, 1, 1])
    Permutation()
    >>> make_permutation([2, 2, 1])
    Permutation(4)(0, 1)(2, 3)
    """
    p = Permutation()
    c = 0
    for j in range(len(partition)):
        a: list[int] = []
        for h in range(partition[j]):
            a.append(c)
            c = c + 1
        if c == 1:
            p1 = Permutation()
            c = 0
        else:
            p1 = Permutation([a])
        p = p * p1
    return p


def size_conjugacy_class(partition: list[int], n: int) -> int:
    """Compute the size of a conjugacy class in a symmetric group.

    .. rubric:: Parameters

    partition : list
        A partition representing the conjugacy class.
    n : int
        The size of the symmetric group.

    .. rubric:: Returns

    int
        The number of elements in the conjugacy class.

    .. rubric:: Examples

    >>> from pyhomrep.utilities import size_conjugacy_class
    >>> size_conjugacy_class([4], 4)
    6
    >>> size_conjugacy_class([1, 1, 1, 1], 4)
    1
    >>> size_conjugacy_class([2, 1, 1], 4)
    6
    >>> size_conjugacy_class([2, 2], 4)
    3
    >>> size_conjugacy_class([3, 1], 4)
    8
    """
    aux1 = 1
    c = 0
    aux = partition[0]
    flag = 1

    for j in range(len(partition)):
        if aux == partition[j]:
            c = c + 1
            flag = 1
        else:
            aux1 = aux1 * (partition[j - 1] ** c) * (math.factorial(c))
            aux = partition[j]
            c = 1
            flag = 0

    if flag == 1:
        aux1 = aux1 * (partition[j - 1] ** c) * (math.factorial(c))
    else:
        aux1 = aux1 * (partition[j] ** c) * (math.factorial(c))

    card = math.factorial(n) / aux1
    return int(card)
