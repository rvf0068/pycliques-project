"""Young tableaux for character computations via the Murnaghan-Nakayama rule.

This module provides the :class:`YoungTableaux` class for computing irreducible
character values of symmetric groups using border-strip tableaux.
"""

from __future__ import annotations

from itertools import permutations


class YoungTableaux:
    """Compute irreducible character values of symmetric groups.

    Uses the Murnaghan-Nakayama rule to compute character values via
    border-strip tableaux. Given two partitions of n, this class computes
    the corresponding character value.

    .. rubric:: Parameters

    p_lambda : list
        A partition represented as a list of positive integers in
        non-increasing order.
    p_rho : list
        Another partition represented as a list of positive integers.

    .. rubric:: Examples

    Compute a character value for S_3::

        >>> from pyhomrep.young import YoungTableaux
        >>> yt = YoungTableaux([3], [1, 1, 1])
        >>> yt.CMNR()
        1
    """

    def __init__(self, p_lambda: list[int], p_rho: list[int]) -> None:
        """Initialize a YoungTableaux instance.

        .. rubric:: Parameters

        p_lambda : list
            A partition representing the shape of the tableaux.
        p_rho : list
            A partition representing the conjugacy class.
        """
        self.p_lambda = p_lambda
        self.p_rho = p_rho

    def choose_tableaux(self, v: list[list[int]]) -> bool:
        """Check if a given list represents a valid border-strip tableaux.

        .. rubric:: Parameters

        v : list
            A list of lists representing a candidate tableaux.

        .. rubric:: Returns

        bool
            True if the list is a valid border-strip tableaux, False otherwise.

        .. rubric:: Examples

        >>> from pyhomrep.young import YoungTableaux
        >>> yt = YoungTableaux([2, 1], [1, 1, 1])
        >>> yt.choose_tableaux([[1, 1, 1], [1, 2]])
        True
        >>> yt.choose_tableaux([[1, 1, 1], [2, 1]])
        False
        """
        # Check rows are non-decreasing
        for row in v:
            for j in range(len(row) - 1):
                if row[j] > row[j + 1]:
                    return False

        # Check columns are non-decreasing
        for i in range(1, len(v)):
            for j in range(len(v[i])):
                if v[i][j] < v[i - 1][j]:
                    return False

        # Check border-strip connectivity
        for i in range(len(v)):
            for j in range(len(v[i])):
                c = 0
                c1 = 0
                if j != 0:
                    if v[i][j] == v[i][j - 1]:
                        c = 1
                        c1 = c1 + 1
                if j != len(v[i]) - 1:
                    if v[i][j] == v[i][j + 1]:
                        c = 1
                        c1 = c1 + 1
                if i != 0:
                    if v[i][j] == v[i - 1][j]:
                        c = 1
                if i != len(v) - 1:
                    if j < len(v[i + 1]):
                        if v[i][j] == v[i + 1][j]:
                            c = 1
                            c1 = c1 + 1
                        if j < len(v[i + 1]) - 1:
                            if v[i][j] == v[i + 1][j + 1]:
                                c1 = c1 + 1
                if (c == 0) and (self.p_rho[v[i][j] - 1] > 1):
                    return False
                if c1 == 3:
                    return False
        return True

    def MNR(self) -> list[list[list[int]]]:
        """Generate all border-strip tableaux for the given partitions.

        .. rubric:: Returns

        list
            A list of border-strip tableaux, where each tableaux is a list of lists.

        .. rubric:: Examples

        >>> from pyhomrep.young import YoungTableaux
        >>> yt = YoungTableaux([4, 1], [3, 2])
        >>> yt.MNR()
        [[[1, 1, 2, 2], [1]]]
        >>> yt2 = YoungTableaux([2, 2, 1], [6])
        >>> yt2.MNR()
        []
        """
        p: list[int] = []
        i = 1
        for h in self.p_rho:
            for _ in range(h):
                p.append(i)
            i = i + 1

        perm = permutations(p)
        d: list[list[list[int]]] = []
        for perm_item in list(perm):
            v: list[int] = list(perm_item)
            c = 0
            w: list[list[int]] = []
            for size in self.p_lambda:
                u: list[int] = []
                for idx in range(c, c + size):
                    u.append(v[idx])
                w.append(u)
                c = c + size
            if self.choose_tableaux(w):
                d.append(w)

        # Remove duplicates
        d1: list[list[list[int]]] = []
        if d:
            d1 = [d[0]]
            for k1 in d:
                if k1 not in d1:
                    d1.append(k1)
        return d1

    def heights(self) -> list[int]:
        """Calculate the heights (sum of heights of border strips).

        .. rubric:: Returns

        list
            A list of height values for each border-strip tableaux.

        .. rubric:: Examples

        >>> from pyhomrep.young import YoungTableaux
        >>> yt = YoungTableaux([5, 2, 1], [3, 3, 1, 1])
        >>> yt.heights()
        [1, 1, 1, 2, 2, 3]
        """
        h_list = self.MNR()
        heights: list[int] = []
        for h in h_list:
            he: list[int] = []
            for i in range(len(self.p_rho)):
                c = 0
                for g in h:
                    if (i + 1) in g:
                        c = c + 1
                he.append(c - 1)
            heights.append(sum(he))
        return heights

    def CMNR(self) -> int:
        """Compute the irreducible character value via Murnaghan-Nakayama rule.

        .. rubric:: Returns

        int
            The irreducible character value of the symmetric group
            for the given partitions.

        .. rubric:: Examples

        >>> from pyhomrep.young import YoungTableaux
        >>> yt = YoungTableaux([5, 2, 1], [3, 3, 1, 1])
        >>> yt.CMNR()
        -2
        """
        heights = self.heights()
        s = 0
        for j in heights:
            s = s + (-1) ** j
        return s
