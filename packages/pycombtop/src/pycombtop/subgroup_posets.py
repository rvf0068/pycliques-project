"""Subgroup posets of finite permutation groups, using SymPy.

Subgroups are labeled by their actual element sets, not abstract isomorphism
types, conjugacy classes, or generating sets. All algorithms here are exact
and intended for small finite groups.
"""

from collections import deque
from collections.abc import Hashable, Iterator
from typing import Literal

from sympy import isprime
from sympy.combinatorics import Permutation, PermutationGroup

from .posets import Poset

Subgroup = frozenset[Permutation]


class SubgroupEnumerationLimit(RuntimeError):
    """An explicit group-order or subgroup-count budget was exceeded."""


def _validate_limit(name: str, value: int | None) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer or None")


def _validate_prime(p: int) -> None:
    if not isinstance(p, int) or isinstance(p, bool) or not isprime(p):
        raise ValueError("p must be a prime integer")


def _p_power(n: int, p: int) -> bool:
    while n % p == 0:
        n //= p
    return n == 1


def enumerate_subgroups(
    group: PermutationGroup,
    *,
    p: int | None = None,
    elementary: bool = False,
    max_group_order: int | None = 1000,
    max_subgroups: int | None = 10_000,
) -> tuple[Subgroup, ...]:
    """Enumerate all subgroups, or only (elementary abelian) p-subgroups.

    Always includes the trivial subgroup. If p is specified, only groups
    of p-power order are returned. With elementary=True they must also be
    abelian of exponent dividing p. Subgroups are found by extending known
    generating sets by one element, skipping redundant right cosets.
    An exceeded budget raises SubgroupEnumerationLimit; no partial list
    is returned. SymPy permutation groups are finite by construction.
    """
    if not isinstance(group, PermutationGroup):
        raise TypeError("expected a SymPy PermutationGroup")
    _validate_limit("max_group_order", max_group_order)
    _validate_limit("max_subgroups", max_subgroups)
    if p is not None:
        _validate_prime(p)
    elif elementary:
        raise ValueError("elementary=True requires a prime p")
    order = int(group.order())
    if max_group_order is not None and order > max_group_order:
        raise SubgroupEnumerationLimit("max_group_order exceeded")
    identity = group.identity
    trivial: Subgroup = frozenset({identity})
    if p is not None and order % p != 0:
        return (trivial,)
    elements = tuple(group.generate_schreier_sims())
    if p is None:
        candidates = elements
    elif elementary:
        candidates = tuple(g for g in elements if int(g.order()) == p)
    else:
        candidates = tuple(g for g in elements if _p_power(int(g.order()), p))

    generators: dict[Subgroup, tuple[Permutation, ...]] = {trivial: ()}
    queue = deque([trivial])
    while queue:
        h = queue.popleft()
        gens = generators[h]
        skipped = set(h)
        for g in candidates:
            if g in skipped:
                continue
            # Every element in H*g generates the same extension with H.
            skipped.update(x * g for x in h)
            if elementary and any(g * x != x * g for x in gens):
                continue
            extension = PermutationGroup([*gens, g])
            if p is not None and not _p_power(int(extension.order()), p):
                continue
            k: Subgroup = frozenset(extension.generate_schreier_sims())
            if k in generators:
                continue
            if max_subgroups is not None and len(generators) >= max_subgroups:
                raise SubgroupEnumerationLimit("max_subgroups exceeded")
            generators[k] = (*gens, g)
            queue.append(k)
    return tuple(generators)


def _subgroup_poset(subgroups: tuple[Subgroup, ...]) -> Poset:
    """Inclusion order, generating only immediate inclusions."""
    ordered = sorted(subgroups, key=len)

    def edges() -> Iterator[tuple[Subgroup, Subgroup]]:
        for i, h in enumerate(ordered):
            covers: list[Subgroup] = []
            for k in ordered[i + 1 :]:
                if len(k) % len(h) or not h < k:
                    continue
                if not any(c <= k for c in covers):
                    covers.append(k)
                    yield h, k

    return Poset(ordered, edges())


def _named_subgroup_poset(
    group: PermutationGroup,
    kind: Literal["lattice", "brown", "quillen"],
    p: int | None,
    proper: bool,
    max_group_order: int | None,
    max_subgroups: int | None,
) -> Poset:
    subgroups = enumerate_subgroups(
        group,
        p=p,
        elementary=kind == "quillen",
        max_group_order=max_group_order,
        max_subgroups=max_subgroups,
    )
    if kind != "lattice":
        subgroups = tuple(h for h in subgroups if len(h) > 1)
    elif proper:
        order = int(group.order())
        subgroups = tuple(h for h in subgroups if 1 < len(h) < order)
    return _subgroup_poset(subgroups)


def subgroup_lattice(
    group: PermutationGroup,
    *,
    proper: bool = False,
    max_group_order: int | None = 1000,
    max_subgroups: int | None = 10_000,
) -> Poset:
    """Lattice of all actual subgroups of group, ordered by inclusion.

    With proper=True remove the trivial subgroup and the whole group.
    Limits are as in enumerate_subgroups.
    """
    return _named_subgroup_poset(
        group, "lattice", None, proper, max_group_order, max_subgroups
    )


def brown_poset(
    group: PermutationGroup,
    p: int,
    *,
    max_group_order: int | None = 1000,
    max_subgroups: int | None = 10_000,
) -> Poset:
    """Brown's S_p(G): all nontrivial p-subgroups ordered by inclusion.

    The whole group is included if it is a nontrivial p-group.
    """
    _validate_prime(p)
    return _named_subgroup_poset(
        group, "brown", p, False, max_group_order, max_subgroups
    )


def quillen_poset(
    group: PermutationGroup,
    p: int,
    *,
    max_group_order: int | None = 1000,
    max_subgroups: int | None = 10_000,
) -> Poset:
    """Quillen's A_p(G): nontrivial elementary abelian p-subgroups."""
    _validate_prime(p)
    return _named_subgroup_poset(
        group, "quillen", p, False, max_group_order, max_subgroups
    )


def subgroup_conjugation(poset: Poset, g: Permutation) -> dict[Subgroup, Subgroup]:
    """The order automorphism H -> g**(-1) H g of a subgroup poset.

    Raises ValueError if the labels are not subgroup element sets, or the
    conjugation does not preserve the given poset. The caller is responsible
    for supplying an element of the desired acting group.
    """
    mapping: dict[Subgroup, Subgroup] = {}
    for h in poset:
        if (
            not isinstance(h, frozenset)
            or not h
            or not all(isinstance(x, Permutation) for x in h)
        ):
            raise ValueError("expected nonempty permutation-set labels")
        # Normalize the conjugator to the permutation degree of the labels.
        degree = next(iter(h)).size
        conjugator = Permutation(g.array_form, size=degree)
        inverse = conjugator**-1
        mapping[h] = frozenset(inverse * x * conjugator for x in h)
    checked: dict[Hashable, Hashable] = {h: image for h, image in mapping.items()}
    if not poset.is_order_preserving(poset, checked):
        raise ValueError("conjugation does not preserve the poset")
    return mapping
