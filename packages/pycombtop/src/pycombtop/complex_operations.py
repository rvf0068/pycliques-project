"""Basic constructions on finite abstract simplicial complexes.

The ambient base set is retained when it matters (notably Alexander duality).
The void complex has no facets; the complex {empty face} is distinct.
"""

from collections.abc import Hashable, Iterable, Mapping

from .posets import face_poset, order_complex
from .simplex import Simplex, SimplicialComplex


def relabel_complex(
    complex_: SimplicialComplex, mapping: Mapping[Hashable, Hashable]
) -> SimplicialComplex:
    """Relabel the entire base set by an injective mapping."""
    if set(mapping) != complex_.base_set or len(set(mapping.values())) != len(mapping):
        raise ValueError("expected an injective relabeling of the entire base set")
    return SimplicialComplex(
        mapping.values(),
        facet_set=({mapping[v] for v in face} for face in complex_.facet_set),
    )


def induced_subcomplex(
    complex_: SimplicialComplex, vertices: Iterable[Hashable]
) -> SimplicialComplex:
    """Restrict to a subset of the ambient base set, including ghost vertices."""
    ground = set(vertices)
    if not ground <= complex_.base_set:
        raise ValueError("restriction contains a point outside the base set")
    return SimplicialComplex(
        ground, facet_set=(face & ground for face in complex_.facet_set)
    )


def simplicial_join(
    first: SimplicialComplex, second: SimplicialComplex
) -> SimplicialComplex:
    """Join two complexes on disjoint base sets; relabel first if necessary."""
    if first.base_set & second.base_set:
        raise ValueError("join requires disjoint base sets")
    return SimplicialComplex(
        first.base_set | second.base_set,
        facet_set=(a | b for a in first.facet_set for b in second.facet_set),
    )


def _fresh_vertices(ground: set, count: int) -> list[int]:
    result: list[int] = []
    candidate = 0
    while len(result) < count:
        if candidate not in ground:
            result.append(candidate)
        candidate += 1
    return result


def cone(
    complex_: SimplicialComplex, apex: Hashable | None = None
) -> SimplicialComplex:
    """Join with a new vertex; None selects an unused nonnegative integer."""
    if apex is None:
        apex = _fresh_vertices(complex_.base_set, 1)[0]
    return simplicial_join(complex_, SimplicialComplex({apex}, [{apex}]))


def suspension(
    complex_: SimplicialComplex, apices: tuple[Hashable, Hashable] | None = None
) -> SimplicialComplex:
    """Unreduced simplicial suspension, the join with two isolated vertices."""
    a: Hashable
    b: Hashable
    if apices is None:
        a, b = _fresh_vertices(complex_.base_set, 2)
    else:
        a, b = apices
    if a == b:
        raise ValueError("suspension apices must be distinct")
    return simplicial_join(complex_, SimplicialComplex({a, b}, [{a}, {b}]))


def wedge_sum(
    first: SimplicialComplex,
    second: SimplicialComplex,
    x: Hashable,
    y: Hashable,
) -> SimplicialComplex:
    """Identify the actual vertices x and y, preserving other labels.

    No other base labels may coincide after identification. Relabel either
    input explicitly to take a wedge of two overlapping copies.
    """
    if x not in first.vertex_set or y not in second.vertex_set:
        raise ValueError("wedge basepoints must be actual vertices")
    if (second.base_set - {y}) & first.base_set:
        raise ValueError("wedge requires disjoint labels away from the basepoints")
    mapping = {v: x if v == y else v for v in second.base_set}
    renamed = relabel_complex(second, mapping)
    return SimplicialComplex(
        first.base_set | renamed.base_set,
        facet_set=first.facet_set | renamed.facet_set,
    )


def face_deletion(
    complex_: SimplicialComplex, sigma: Iterable[Hashable]
) -> SimplicialComplex:
    """Remove all faces containing sigma, without removing ambient labels.

    For a singleton this agrees with vertex deletion after retaining the
    deleted vertex as a ghost. Deleting the empty face yields the void
    complex. A nonface deletion has no effect.
    """
    face = frozenset(sigma)
    if not face <= complex_.base_set:
        raise ValueError("face outside the base set")
    facets: list[Simplex] = []
    for f in complex_.facet_set:
        if face <= f:
            facets.extend(Simplex(f - {v}) for v in face)
        else:
            facets.append(Simplex(f))
    return SimplicialComplex(complex_.base_set, facet_set=facets)


def barycentric_subdivision(complex_: SimplicialComplex) -> SimplicialComplex:
    """Order complex of the nonempty face poset (ghost vertices disappear)."""
    if not complex_.facet_set:
        return SimplicialComplex(set(), facet_set=[])
    return order_complex(face_poset(complex_))


def minimal_nonfaces(complex_: SimplicialComplex) -> set[Simplex]:
    """Minimal subsets of the ambient base set not belonging to the complex.

    Compute minimal transversals of the complements of the facets, using
    incremental antichain pruning. This may have exponential output.
    The void complex has the empty set as its unique minimal nonface.
    """
    candidates = {Simplex([])}
    for facet in complex_.facet_set:
        missing = complex_.base_set - facet
        extended = {
            Simplex(c | {v}) for c in candidates if not c & missing for v in missing
        }
        extended.update(c for c in candidates if c & missing)
        candidates = set()
        for c in sorted(extended, key=len):
            if not any(d <= c for d in candidates):
                candidates.add(c)
        if not candidates:
            break
    return candidates


def alexander_dual(complex_: SimplicialComplex) -> SimplicialComplex:
    """Alexander dual relative to the entire ambient base set.

    A face A belongs to the dual exactly when its complement is not a face
    of the input. Dual facets are complements of minimal nonfaces.
    """
    return SimplicialComplex(
        complex_.base_set,
        facet_set=(complex_.base_set - f for f in minimal_nonfaces(complex_)),
    )
