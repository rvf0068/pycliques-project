"""Decomposition of representations into irreducibles.

This module provides functions for decomposing matrix representations into
their irreducible components using standard representation theory techniques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp
from sympy.matrices import Matrix

from pyhomrep.matrix import MatrixRepresentation

if TYPE_CHECKING:
    from sympy.matrices import ImmutableMatrix
    from sympy.matrices.dense import MutableDenseMatrix


def block(m: Matrix) -> list[int]:
    """Find the indices where matrix blocks end.

    For a block diagonal matrix (or Jordan form), this function returns
    the indices of the last columns of each block.

    .. rubric:: Parameters

    m : Matrix
        A sympy square matrix.

    .. rubric:: Returns

    list
        A list of column indices where blocks end.

    .. rubric:: Examples

    >>> from sympy.matrices import Matrix
    >>> from pyhomrep.decomposition import block
    >>> M = Matrix([
    ...     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    >>> block(M)
    [0, 4, 5, 6, 7, 8, 9]
    """
    v: list[int] = []
    c1 = 0
    i = 0
    n = m.shape[0]
    while c1 < n:
        c = 0
        for j in range(c1, n):
            if m[i, j] != 0 or m[j, i] != 0:
                if sp.Abs(i - j) > c:
                    c = sp.Abs(i - j)
        if c == 0:
            v.append(c1)
            c1 = c1 + 1
            i = c1
        else:
            bloques = False
            while not bloques:
                bloques = True
                for j in range(c1, c1 + c + 1):
                    for k in range(c1 + c + 1, n):
                        if m[j, k] != 0 or m[k, j] != 0:
                            if sp.Abs(i - k) > c:
                                c = sp.Abs(i - k)
            v.append(c1 + c)
            c1 = c1 + c + 1
            i = c1
    return v


def blockI(m: Matrix, n: int, i: int) -> MutableDenseMatrix:
    """Insert a matrix at position (i, i) of an identity matrix.

    .. rubric:: Parameters

    m : Matrix
        The matrix to insert.
    n : int
        The size of the resulting matrix.
    i : int
        The row/column index where to insert the matrix.

    .. rubric:: Returns

    Matrix
        An n×n identity matrix with m inserted at position (i, i).

    .. rubric:: Examples

    >>> from sympy.matrices import Matrix
    >>> from pyhomrep.decomposition import blockI
    >>> M = Matrix([[1, 1, 1], [1, 1, 1]])
    >>> blockI(M, 4, 0)
    Matrix([
    [1, 1, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])
    """
    result = sp.eye(n)
    for j in range(m.shape[0]):
        for k in range(m.shape[1]):
            result[j + i, k + i] = m[j, k]
    return result


def MTS(a: Matrix) -> MutableDenseMatrix:
    """Create a triangular matrix V such that V^H * A * V = I.

    For a positive definite Hermitian matrix A, this computes an upper
    triangular matrix V satisfying V^H * A * V = I (a form of Cholesky
    decomposition).

    .. rubric:: Parameters

    a : Matrix
        A positive definite Hermitian matrix.

    .. rubric:: Returns

    Matrix
        An upper triangular matrix V such that V^H * A * V = I.

    .. rubric:: Examples

    >>> from sympy.matrices import Matrix
    >>> M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
    >>> A = M.H * M
    >>> from pyhomrep.decomposition import MTS
    >>> V = MTS(A)
    >>> V.H * A * V
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    """
    a1 = a
    n = a.shape[0]
    v = sp.eye(n)
    for i in range(n):
        c = sp.eye(n)
        c[i, i] = 1 / sp.sqrt(a1[i, i])
        for j in range(i + 1, n):
            c[i, j] = -(1 / a1[i, i]) * a1[i, j]
        v = v * c
        a1 = (c.H) * a1 * c
    return v


def unitary_representation(d: MatrixRepresentation) -> MatrixRepresentation:
    """Convert a representation to a unitary representation.

    .. rubric:: Parameters

    d : MatrixRepresentation
        A matrix representation.

    .. rubric:: Returns

    MatrixRepresentation
        An equivalent unitary matrix representation.
    """
    G = d.group
    n = d.degree
    A = sp.zeros(n, n)
    for g in d._elements:
        J = (d.map[g].H) * d.map[g]
        J = sp.expand(J)
        A = J + A
    V = MTS(A)
    M = {g: sp.ImmutableMatrix((V.inv()) * d.map[g] * V) for g in d._elements}
    return MatrixRepresentation(M, G, n)


def is_irreducible(d: MatrixRepresentation) -> bool | Matrix:
    """Determine if a representation is irreducible.

    .. rubric:: Parameters

    d : MatrixRepresentation
        A matrix representation.

    .. rubric:: Returns

    bool or Matrix
        True if the representation is irreducible; otherwise, a matrix
        that can be used to reduce it.
    """
    n = d.degree
    N = sp.eye(n)
    R = unitary_representation(d)
    for r in range(n):
        for s in range(n):
            H = sp.zeros(n)
            if n - 1 - r == n - 1 - s:
                H[n - 1 - r, n - 1 - r] = 1
            else:
                if n - 1 - r > n - 1 - s:
                    H[n - 1 - r, n - 1 - s] = 1
                    H[n - 1 - s, n - 1 - r] = 1
                else:
                    H[n - 1 - r, n - 1 - s] = 1 * sp.I
                    H[n - 1 - s, n - 1 - r] = -1 * sp.I
            M: Any = sp.zeros(n)
            for g in R._elements:
                M = M + (R.map[g].H * H * R.map[g])
            M = (sp.sympify(1) / n) * M
            M = sp.expand(M)
            if M != M[0, 0] * N:
                return M
    return True


def reduce(d: MatrixRepresentation) -> ImmutableMatrix:
    """Decompose a representation into irreducibles.

    .. rubric:: Parameters

    d : MatrixRepresentation
        The representation to decompose.

    .. rubric:: Returns

    Matrix
        A matrix U which decomposes the representation.
    """
    G = d.group
    b = d.degree
    M = is_irreducible(d)
    if M is True:
        return sp.ImmutableMatrix(sp.eye(b))
    else:
        (P, _J) = M.jordan_form()
        P = sp.expand(P)
        w = [block(P.inv() * d.map[g] * P) for g in d._elements]
        length = len(w[0])
        au = w[0]
        for g in w:
            if len(g) < length:
                length = len(g)
                au = g
        e = 0
        U = P
        for a in au:
            d1 = {
                g: sp.ImmutableMatrix((P.inv() * d.map[g] * P)[e : a + 1, e : a + 1])
                for g in d._elements
            }
            U = U * blockI(reduce(MatrixRepresentation(d1, G, (a + 1 - e))), b, e)
            e = a + 1
        return sp.ImmutableMatrix(U)
