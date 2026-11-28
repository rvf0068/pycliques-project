"""Finite permutation actions on labeled finite sets and combinatorial structures.

The acting group is represented by SymPy's exact permutation-group backend.
A permutation acts on positions in the supplied elements sequence, so
labels may be arbitrary hashable Python objects.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING

from sympy.combinatorics import Permutation, PermutationGroup

if TYPE_CHECKING:
    from .posets import Poset
    from .simplex import SimplicialComplex


class PermutationAction:
    """A finite permutation group acting on an explicitly ordered finite set.

    generators may be SymPy permutations of positions or a SymPy
    PermutationGroup. The i-th position corresponds to elements[i]. Labels
    may be arbitrary hashable Python objects. The group is never enumerated.

    The order of elements is part of the representation: use the same order
    when comparing two actions or passing an action to another package.
    """

    def __init__(
        self,
        elements: Iterable[Hashable],
        generators: Iterable[Permutation] | PermutationGroup,
    ) -> None:
        self.elements = tuple(elements)
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("action elements must be distinct")
        self.index = {element: i for i, element in enumerate(self.elements)}
        degree = len(self.elements)

        raw = (
            generators.generators
            if isinstance(generators, PermutationGroup)
            else generators
        )
        normalized: list[Permutation] = []
        for generator in raw:
            if not isinstance(generator, Permutation):
                raise TypeError("generators must be SymPy Permutation objects")
            if generator.size > degree:
                raise ValueError("generator degree exceeds the action-set size")
            normalized.append(Permutation(generator.array_form, size=degree))

        self.generators = tuple(normalized)
        if not self.generators:
            self.group = PermutationGroup([Permutation([], size=degree)])
        else:
            self.group = PermutationGroup(self.generators)

    def __len__(self) -> int:
        return len(self.elements)

    def __contains__(self, element: object) -> bool:
        return element in self.index

    def image(self, element: Hashable, permutation: Permutation) -> Hashable:
        """Apply a group element, rejecting permutations outside this group."""
        if element not in self.index:
            raise KeyError(element)
        if not isinstance(permutation, Permutation):
            raise TypeError("permutation must be a SymPy Permutation")
        if permutation.size > len(self.elements):
            raise ValueError("permutation degree exceeds the action-set size")
        normalized = Permutation(permutation.array_form, size=len(self.elements))
        if not self.group.contains(normalized):
            raise ValueError("permutation is not in the acting group")
        return self.elements[normalized(self.index[element])]

    def orbit(self, element: Hashable) -> tuple[Hashable, ...]:
        """Return the orbit in the input order of elements."""
        if element not in self.index:
            raise KeyError(element)
        indices = self.group.orbit(self.index[element])
        return tuple(x for i, x in enumerate(self.elements) if i in indices)

    def orbits(self) -> tuple[tuple[Hashable, ...], ...]:
        """Return all orbits, ordered by their first input position."""
        parts = self.group.orbits()
        return tuple(
            tuple(self.elements[i] for i in sorted(part))
            for part in sorted(parts, key=min)
        )

    def stabilizer(self, element: Hashable) -> PermutationGroup:
        """Return the SymPy subgroup fixing element (in index coordinates)."""
        if element not in self.index:
            raise KeyError(element)
        return self.group.stabilizer(self.index[element])

    def fixed_points(self) -> tuple[Hashable, ...]:
        """Return points fixed by the entire acting group."""
        return tuple(
            element
            for i, element in enumerate(self.elements)
            if all(generator(i) == i for generator in self.generators)
        )

    def is_invariant(self, subset: Iterable[Hashable]) -> bool:
        """Whether a subset is preserved setwise by every generator."""
        chosen = set(subset)
        if not chosen <= self.index.keys():
            return False
        return all(
            {self.elements[g(self.index[x])] for x in chosen} == chosen
            for g in self.generators
        )

    @property
    def vertex_maps(self):
        """Generator maps on labels, suitable for pyhomrep.OrbitChainComplex."""
        maps = []
        for generator in self.generators:

            def apply(element, permutation=generator):
                if element not in self.index:
                    raise KeyError(element)
                return self.elements[permutation(self.index[element])]

            maps.append(apply)
        return tuple(maps)

    def validate_poset(self, poset: Poset) -> None:
        """Raise ValueError unless the permutations are poset automorphisms."""
        if set(self.elements) != set(poset):
            raise ValueError("action elements must equal the poset elements")
        for generator in self.generators:
            mapping = {
                x: self.elements[generator(i)] for i, x in enumerate(self.elements)
            }
            if any(
                poset.leq(a, b) != poset.leq(mapping[a], mapping[b])
                for a in poset
                for b in poset
            ):
                raise ValueError("each generator must be a poset automorphism")

    def validate_complex(self, complex_: SimplicialComplex) -> None:
        """Raise ValueError unless the permutations are simplicial automorphisms."""
        if set(self.elements) != complex_.vertex_set:
            raise ValueError("action elements must equal the complex vertex set")
        facets = complex_.facet_set
        for generator in self.generators:
            image_facets = {
                frozenset(self.elements[generator(self.index[x])] for x in facet)
                for facet in facets
            }
            if image_facets != facets:
                raise ValueError("each generator must preserve the simplicial complex")

    def validate_invariant_subcomplex(
        self, ambient: SimplicialComplex, subcomplex: SimplicialComplex
    ) -> None:
        """Raise ValueError unless subcomplex is an invariant subcomplex."""
        ambient_faces = ambient.all_simplices()
        if not subcomplex.vertex_set <= ambient.vertex_set or any(
            facet not in ambient_faces for facet in subcomplex.facet_set
        ):
            raise ValueError("relative complex must be a subcomplex")
        for generator in self.generators:
            image_facets = {
                frozenset(self.elements[generator(self.index[x])] for x in facet)
                for facet in subcomplex.facet_set
            }
            if image_facets != subcomplex.facet_set:
                raise ValueError("relative complex must be invariant under the action")
