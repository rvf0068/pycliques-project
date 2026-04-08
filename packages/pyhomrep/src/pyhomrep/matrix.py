"""Matrix representations of groups.

This module provides the :class:`MatrixRepresentation` class for working with
matrix representations of finite groups, particularly permutation groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp
from sympy.combinatorics import Permutation

if TYPE_CHECKING:
    from sympy.combinatorics.perm_groups import PermutationGroup
    from sympy.matrices import ImmutableMatrix


class MatrixRepresentation:
    """A matrix representation of a finite group.

    Matrix representations map group elements to invertible matrices in a way
    that preserves group structure (i.e., a group homomorphism).

    .. rubric:: Parameters

    d : dict
        A dictionary mapping group elements to matrices.
    G : PermutationGroup
        The group being represented.
    n : int
        The degree (dimension) of the representation.

    .. rubric:: Examples

    Create a regular representation::

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from pyhomrep.matrix import regular_representation
        >>> rr = regular_representation(SymmetricGroup(3))
        >>> rr.degree
        6
    """

    def __init__(
        self,
        d: dict[Permutation, ImmutableMatrix],
        G: PermutationGroup,
        n: int,
    ) -> None:
        """Initialize a matrix representation.

        .. rubric:: Parameters

        d : dict
            A dictionary mapping group elements to matrices.
        G : PermutationGroup
            The group being represented.
        n : int
            The degree (dimension) of the representation.
        """
        self.map = d
        self.group = G
        self.degree = n
        self._elements = tuple(G.generate())

    def character(self) -> dict[Permutation, Any]:
        """Compute the character of the representation.

        The character is a function from the group to the base field that
        assigns to each group element the trace of its corresponding matrix.

        .. rubric:: Returns

        dict
            A dictionary mapping group elements to their character values
            (traces of matrices).

        .. rubric:: Examples

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from pyhomrep.matrix import regular_representation
        >>> rr = regular_representation(SymmetricGroup(3))
        >>> char = rr.character()
        >>> char[Permutation(2)]  # identity element
        6
        """
        return {g: self.map[g].trace() for g in self._elements}

    def is_unitary(self) -> bool:
        """Check if the representation is unitary.

        A representation is unitary if all matrices satisfy M^H * M = I,
        where M^H is the conjugate transpose.

        .. rubric:: Returns

        bool
            True if the representation is unitary, False otherwise.

        .. rubric:: Examples

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from pyhomrep.matrix import regular_representation
        >>> rr = regular_representation(SymmetricGroup(3))
        >>> rr.is_unitary()
        True
        """
        for g in self._elements:
            if sp.expand(self.map[g].H * self.map[g]) != sp.eye(self.degree):
                return False
        return True


def _char_f(elems: tuple, g: Permutation, i: int, j: int) -> int:
    """Helper function for regular representation matrix entries."""
    if elems[i] * g == elems[j]:
        return 1
    return 0


def regular_representation(G: PermutationGroup) -> MatrixRepresentation:
    """Build the regular representation of a group.

    The regular representation has dimension equal to the order of the group.
    Each group element acts by permuting a basis indexed by the group elements
    themselves.

    .. rubric:: Parameters

    G : PermutationGroup
        The group to represent.

    .. rubric:: Returns

    MatrixRepresentation
        The regular representation of G.

    .. rubric:: Examples

    >>> from sympy.combinatorics.named_groups import SymmetricGroup
    >>> from pyhomrep.matrix import regular_representation
    >>> rr = regular_representation(SymmetricGroup(3))
    >>> rr.degree
    6
    >>> type(rr)
    <class 'pyhomrep.matrix.MatrixRepresentation'>
    """
    n = G.order()
    elts = tuple(G.generate())
    mydict = {
        g: sp.ImmutableMatrix(sp.Matrix(n, n, lambda i, j: _char_f(elts, g, i, j)))
        for g in elts
    }
    return MatrixRepresentation(mydict, G, n)
