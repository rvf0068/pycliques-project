"""Reference elementary free-face collapse implementation."""

from .simplex import Simplex, SimplicialComplex, _stable_vertex_key


def _is_free_face(s_complex: SimplicialComplex, simplex: Simplex) -> bool:
    """An elementary free face has a unique facet, one dimension higher."""
    if not simplex or simplex in s_complex.facet_set:
        return False
    containing = [f for f in s_complex.facet_set if simplex.issubset(f)]
    return len(containing) == 1 and len(containing[0]) == len(simplex) + 1


def _remove_simplex(
    s_complex: SimplicialComplex, simplex: Simplex
) -> SimplicialComplex:
    """Delete exactly one elementary free face/coface pair.

    All other faces survive, notably the other codimension-one faces of
    the removed facet. The result retains only inclusion-maximal facets.
    """
    if not _is_free_face(s_complex, simplex):
        raise ValueError("Not a codimension-one elementary free face")
    facet = next(f for f in s_complex.facet_set if simplex.issubset(f))
    survivors = s_complex.facet_set - {facet}
    added = (Simplex(facet - {vertex}) for vertex in facet)
    facets = survivors | {
        face
        for face in added
        if face != simplex and not any(face <= other for other in survivors)
    }
    return SimplicialComplex._from_maximal_faces(set().union(*facets), facets)


def _has_free_face(s_complex: SimplicialComplex) -> Simplex | None:
    """Search only codimension-one faces of facets, not all simplices."""
    for facet in sorted(
        s_complex.facet_set,
        key=lambda s: (len(s), tuple(sorted(_stable_vertex_key(v) for v in s))),
        reverse=True,
    ):
        for vertex in sorted(facet, key=_stable_vertex_key):
            face = Simplex(facet - {vertex})
            if _is_free_face(s_complex, face):
                return face
    return None


def collapse(s_complex: SimplicialComplex) -> SimplicialComplex:
    """Greedily apply *genuine* codimension-one elementary collapses.

    Each removal preserves simple-homotopy type. Failure to find another
    free face is not a non-collapsibility certificate [Whitehead1939].
    """
    sc = s_complex
    while len(sc.vertex_set) > 1:
        if len(sc.facet_set) == 1:
            # A simplex has an explicit elementary collapse to a point,
            # but enumerating its 2^(n-1) pairs is unnecessary.
            one_vertex = min(sc.vertex_set, key=_stable_vertex_key)
            return SimplicialComplex({one_vertex}, facet_set=[{one_vertex}])
        free_face = _has_free_face(sc)
        if free_face is None:
            break
        sc = _remove_simplex(sc, free_face)
    return sc
