"""
Homotopy type computation for clique complexes of graphs.

This module provides a :class:`HomotopyVerdict` dataclass that records both the
homotopy type of a simplicial complex and the theorem(s) used to establish it,
together with a cascade of strategies (:func:`homotopy_type_with_verdict`) that
tries each technique in turn and returns as soon as one succeeds.

Dependencies
------------
- networkx
- sympy
- mogutda
- pycliques (for :mod:`pycliques.dominated`, :mod:`pycliques.surfaces`)

References
----------
.. [Dong2003] X. Dong, "Topology of bounded-degree graph complexes",
   J. Algebra 262 (2003), 287--312. DOI: 10.1016/S0021-8693(03)00024-3.
   (Vertex-ordered discrete Morse matching.)

.. [Forman1998] R. Forman, "Morse theory for cell complexes",
   Adv. Math. 134 (1998), 90--145. DOI: 10.1006/aima.1997.1650.

.. [BW1996] A. Björner and M. L. Wachs,
   "Shellable nonpure complexes and posets. I",
   Trans. Amer. Math. Soc. 348 (1996), 1299--1327.

.. [BFJ2010] R. Boulet, E. Fieux and B. Jouve,
   "Simplicial simple-homotopy of flag complexes in terms of graphs",
   European J. Combin. 31 (2010), 161--176.
   DOI: 10.1016/j.ejc.2009.05.003. (Graph s-homotopy.)

.. [Whitehead1939] J. H. C. Whitehead,
   "Simplicial spaces, nuclei and m-groups",
   Proc. London Math. Soc. (2) 45 (1939), 243--327.

.. [FríasArmenta2013] M. E. Frías-Armenta, F. Larrión,
   V. Neumann-Lara and M. A. Pizaña,
   "Edge contraction and edge removal on iterated clique graphs",
   Discrete Appl. Math. 161 (2013), 1427--1439.
   DOI: 10.1016/j.dam.2013.02.003. (Local cutpoints.)

The graph-join, cutvertex-wedge, link-attachment and edge-attachment
reductions below follow directly from elementary simplicial constructions
and homotopy pushouts, not from Mayer--Vietoris homology alone.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import cast

import mogutda
import networkx as nx
from pycliques import clique_graph
from pycliques.surfaces import open_neighborhood

from .s_collapses import complete_s_collapse, complete_s_collapse_edges
from .simplex import Simplex, SimplicialComplex, clique_complex, dong_matching
from .simplex import star as star  # Public re-export

# ---------------------------------------------------------------------------
# Theorem citations
# ---------------------------------------------------------------------------


class Theorem:
    """Human-readable mathematical justifications, not bare theorem names."""

    CONTRACTIBLE_POINT = "A one-vertex clique complex is a point."
    S_COLLAPSE_TO_POINT = (
        "Graph s-collapse reduction to a point: s-dismantlable vertex and "
        "edge removals preserve the simple-homotopy type of the clique "
        "complex [BFJ2010]."
    )
    DONG_MATCHING = (
        "Dong's vertex-ordered matching is checked to be acyclic on the "
        "augmented face poset. Its critical nonempty faces all have one "
        "dimension, so Forman's discrete Morse CW model is a wedge of "
        "spheres [Dong2003, Forman1998]."
    )
    VERTEX_DECOMPOSABLE = (
        "A checked shedding-vertex recursion certifies nonpure vertex "
        "decomposability. The complex is shellable, hence a wedge of "
        "spheres, with multiplicities determined by reduced Betti "
        "numbers [BW1996]."
    )
    JOIN_COMPLEMENT = (
        "A disconnected complement decomposes the graph as a join. "
        "The clique complex is the simplicial join of the certified "
        "factors; the join-suspension identity determines its spheres."
    )
    UNIVERSAL_VERTEX_CONE = (
        "A vertex adjacent to every other vertex is an apex of every "
        "maximal simplex; its clique complex is a cone."
    )
    DISJOINT_COMPLETE_NEIGHBORHOOD = (
        "The open neighborhood is a disjoint union of r>=2 nonempty "
        "complete graphs, and deleting the vertex leaves a connected "
        "graph. The vertex star is a cone attached along r disjoint "
        "contractible simplices, producing r-1 additional circles."
    )
    INDEPENDENT_NEIGHBORHOOD_CLIQUE_GRAPH = (
        "The vertex has r>=2 pairwise nonadjacent neighbors and its "
        "deletion is connected. In the clique graph, the r new edge-cliques "
        "form a simplex; collapsing this simplex turns their attachments "
        "into a cone over r disjoint simplices, adding r-1 circles "
        "[FríasArmenta2013 gives related local-cutpoint terminology]."
    )
    TRIANGLE_FREE_INTERCOMPONENT_EDGES = (
        "Deleting all selected edges leaves precisely two connected "
        "induced pieces, with every deleted edge joining the pieces "
        "and lying in no triangle. The clique complex attaches m "
        "one-cells between two certified pieces, adding m-1 circles."
    )
    CUTVERTEX_WEDGE = (
        "The graph has an articulation vertex. Its clique complex is "
        "the union of the certified complexes of the cutvertex pieces "
        "meeting exactly at that vertex, hence their wedge."
    )
    CONTRACTIBLE_LINK_COMPONENTS = (
        "After a homotopy-preserving reduction, deleting a vertex leaves "
        "a connected complex and the vertex link is a disjoint union "
        "of r>=2 certified contractible complexes. Attaching the cone "
        "on this link adds r-1 circles."
    )
    BETTI_ONLY = (
        "No certified wedge decomposition succeeded; reduced Betti "
        "numbers alone do not determine the homotopy type."
    )
    LARGE_GRAPH_BETTI = (
        "Graph too large for the combinatorial shortcuts; reduced "
        "Betti numbers computed via mogutda, not a homotopy certificate."
    )
    LIMIT_EXCEEDED = "Configured size limit exceeded; no homotopy type established."


# ---------------------------------------------------------------------------
# WedgeOfSpheres  - structured internal representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WedgeOfSpheres:
    """A wedge (one-point union) of spheres, stored as a dimension->count map.

    This is the internal structured representation used during homotopy-type
    computations.  It records how many copies of each sphere dimension appear
    in the wedge.  Convert to a human-readable LaTeX string with
    :meth:`to_latex`.

    .. rubric:: Examples

    >>> from pycombtop.homotopy_type import WedgeOfSpheres
    >>> w = WedgeOfSpheres({1: 3, 2: 1})
    >>> w.to_latex()
    '\\\\(\\\\vee_{3}S^{1}\\\\vee S^{2}\\\\)'
    >>> WedgeOfSpheres.contractible().is_contractible()
    True
    """

    spheres: dict[int, int] = field(default_factory=dict)
    """Mapping from dimension to number of spheres of that dimension."""

    @staticmethod
    def contractible() -> WedgeOfSpheres:
        """Return the contractible (empty wedge) space."""
        return WedgeOfSpheres({})

    @staticmethod
    def sphere(dim: int) -> WedgeOfSpheres:
        """Return a single sphere of the given dimension."""
        return WedgeOfSpheres({dim: 1})

    @staticmethod
    def from_betti(bettis: list[int]) -> WedgeOfSpheres:
        """Build a wedge of spheres from reduced Betti numbers.

        For a space known (by theorem) to have the homotopy type of a
        wedge of spheres, the reduced Betti number in dimension *d*
        equals the number of *d*-spheres in the wedge.
        """
        spheres: dict[int, int] = {}
        for dim, count in enumerate(bettis):
            if count > 0:
                spheres[dim] = count
        return WedgeOfSpheres(spheres)

    def is_contractible(self) -> bool:
        """Return ``True`` if this represents the contractible space."""
        return all(c == 0 for c in self.spheres.values())

    def add_spheres(self, dim: int, count: int = 1) -> WedgeOfSpheres:
        """Return a new wedge with *count* extra spheres of dimension *dim*."""
        new = dict(self.spheres)
        new[dim] = new.get(dim, 0) + count
        return WedgeOfSpheres(new)

    def wedge(self, other: WedgeOfSpheres) -> WedgeOfSpheres:
        """Return the wedge (sum) of this space with *other*."""
        new = dict(self.spheres)
        for dim, count in other.spheres.items():
            new[dim] = new.get(dim, 0) + count
        return WedgeOfSpheres(new)

    def suspend(self) -> WedgeOfSpheres:
        """Return the suspension (shift all dimensions up by 1)."""
        return WedgeOfSpheres({d + 1: c for d, c in self.spheres.items()})

    def to_latex(self) -> str:
        """Convert to a LaTeX wedge-of-spheres string."""
        if self.is_contractible():
            return "Contractible"
        parts = []
        for dim in sorted(self.spheres):
            count = self.spheres[dim]
            if count <= 0:
                continue
            sphere = f"S^{{{dim}}}"
            if count == 1:
                parts.append(sphere)
            else:
                parts.append(f"\\vee_{{{count}}}{sphere}")
        return "\\(" + "\\vee ".join(parts) + "\\)"

    def to_betti(self) -> list[int]:
        """Return reduced Betti numbers as a list."""
        if not self.spheres:
            return []
        max_dim = max(self.spheres)
        result = [0] * (max_dim + 1)
        for dim, count in self.spheres.items():
            result[dim] = count
        # Trim trailing zeros
        while result and result[-1] == 0:
            result.pop()
        return result


# ---------------------------------------------------------------------------
# HomotopyVerdict
# ---------------------------------------------------------------------------


@dataclass
class HomotopyVerdict:
    """The outcome of a homotopy-type computation.

    .. rubric:: Attributes

    wedge : WedgeOfSpheres | None
        For an exact verdict, the established wedge-of-spheres homotopy type.
        For an inexact verdict, this may contain only the reduced Betti-number
        vector encoded as sphere counts; it is not then a homotopy
        decomposition.
    reason : str
        The theorem or sequence of reductions used to establish the
        verdict.  One of the :class:`Theorem` constants, or a
        concatenation of several.
    is_exact : bool
        ``True`` only when the claimed homotopy decomposition is proved.
        Accurately computed Betti numbers alone do not prove a
        wedge-of-spheres decomposition; in particular, an empty ``wedge``
        when ``is_exact`` is false does not prove contractibility.
    is_limited : bool
        ``True`` when computation was skipped because a configured size limit
        was exceeded. A limited result is inconclusive, not contractible.
    """

    wedge: WedgeOfSpheres
    reason: str
    is_exact: bool = True
    is_limited: bool = False

    @property
    def verdict(self) -> str:
        """A human-readable (LaTeX-compatible) string for the homotopy type."""
        if self.is_limited:
            return "Too large to check"
        return self.wedge.to_latex()

    def __str__(self) -> str:  # pragma: no cover
        exact_tag = ""
        if self.is_limited:
            exact_tag = " [inconclusive]"
        elif not self.is_exact:
            exact_tag = " [Betti numbers only]"
        return f"{self.verdict}{exact_tag}  ({self.reason})"


# ---------------------------------------------------------------------------
# Internal helpers  (previously scattered across homsmall.py / homdiv.py)
# ---------------------------------------------------------------------------


def _simplify_graph_ht(graph: nx.Graph) -> nx.Graph:
    """Apply vertex and edge s-collapses to simplify a graph for homotopy."""
    g = complete_s_collapse(graph)
    g = complete_s_collapse_edges(g)
    return complete_s_collapse(g)


def _shuff(lst: set) -> list:
    return random.sample(list(lst), len(lst))


def _too_large_simplicial_complex(
    s_complex: SimplicialComplex,
    max_vertices: int | None,
    max_simplices: int | None,
) -> bool:
    """Return whether a complex exceeds either configured size limit."""
    if max_simplices is not None and max_simplices < 0:
        raise ValueError("max_simplices must be nonnegative or None")
    if max_vertices is not None and len(s_complex.vertex_set) > max_vertices:
        return True
    if max_simplices is None:
        return False
    upper_bound = sum(2 ** len(facet) for facet in s_complex.facet_set)
    if upper_bound <= max_simplices:
        return False
    distinct_faces: set[frozenset] = set()
    for facet in s_complex.facet_set:
        for size in range(len(facet) + 1):
            for face in combinations(facet, size):
                distinct_faces.add(frozenset(face))
                if len(distinct_faces) > max_simplices:
                    return True
    return False


def _graph_exceeds_simplex_budget(graph: nx.Graph, max_simplices: int) -> bool:
    """Check a flag complex's face count while streaming maximal cliques.

    This avoids first constructing a potentially huge facet collection.
    The empty face is counted, exactly as in
    :func:`_too_large_simplicial_complex`. Stop at the first confirmed
    budget violation; overlap between facets is counted only once.
    """
    if max_simplices < 0:
        raise ValueError("max_simplices must be nonnegative or None")
    faces: set[frozenset] = set()
    for clique in nx.find_cliques(graph):
        if (1 << len(clique)) > max_simplices:
            # A single facet already contains too many distinct faces.
            return True
        for size in range(len(clique) + 1):
            for face in combinations(clique, size):
                faces.add(frozenset(face))
                if len(faces) > max_simplices:
                    return True
    # The empty graph has the empty face. It has no maximal cliques.
    return max_simplices == 0


def _limited_verdict() -> HomotopyVerdict:
    """Return the standard inconclusive result for a size-limited input."""
    return HomotopyVerdict(
        wedge=WedgeOfSpheres.contractible(),
        reason=Theorem.LIMIT_EXCEEDED,
        is_exact=False,
        is_limited=True,
    )


# --- Simplicial-complex collapse (elementary collapses) ---


def _facets_containing(s_complex: SimplicialComplex, simplex: Simplex) -> int:
    return sum(1 for f in s_complex.facet_set if simplex.issubset(f))


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
    candidates = set(s_complex.facet_set - {facet})
    for vertex in facet:
        codim_one_face = Simplex(facet - {vertex})
        if codim_one_face != simplex:
            candidates.add(codim_one_face)
    facets = {f for f in candidates if not any(f < other for other in candidates)}
    vertices = set().union(*(set(f) for f in facets))
    return SimplicialComplex(vertices, facet_set=facets)


def _has_free_face(s_complex: SimplicialComplex) -> Simplex | None:
    """Search only codimension-one faces of facets, not all simplices."""
    for facet in sorted(
        s_complex.facet_set,
        key=lambda s: (len(s), repr(sorted(s, key=repr))),
        reverse=True,
    ):
        for vertex in sorted(facet, key=repr):
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
            one_vertex = next(iter(sc.vertex_set))
            return SimplicialComplex({one_vertex}, facet_set=[{one_vertex}])
        free_face = _has_free_face(sc)
        if free_face is None:
            break
        sc = _remove_simplex(sc, free_face)
    return sc


# --- Betti number helpers ---


def _simplify_betti_list(bettis: list[int]) -> list[int]:
    result = list(bettis)
    while result and result[-1] == 0:
        result.pop()
    return result


def betti_numbers_sc(s_complex: SimplicialComplex) -> list[int]:
    """Reduced Betti numbers of *s_complex* using mogutda."""
    simplices = [tuple(c) for c in s_complex.facet_set]
    mc = mogutda.SimplicialComplex(simplices=simplices)
    dim = max((len(f) - 1 for f in s_complex.facet_set), default=-1)
    if dim < 0:
        return []
    numbers = [int(mc.betti_number(i)) for i in range(dim + 1)]
    numbers[0] -= 1  # reduced beta_0
    return _simplify_betti_list(numbers)


def betti_numbers_graph(graph: nx.Graph) -> list[int]:
    """Reduced Betti numbers of the clique complex of *graph* via mogutda."""
    simplices = [tuple(c) for c in nx.find_cliques(graph)]
    mc = mogutda.SimplicialComplex(simplices=simplices)
    dim = max((len(s) - 1 for s in simplices), default=-1)
    numbers = [int(mc.betti_number(i)) for i in range(dim + 1)]
    numbers[0] -= 1
    return _simplify_betti_list(numbers)


def _read_dong(dong: set[Simplex]) -> WedgeOfSpheres | None:
    """Read a *verified acyclic augmented* Dong matching.

    The empty face must have been paired, and all remaining critical
    faces must have the same positive cardinality. The Morse CW model
    then has one 0-cell and n cells of a common positive dimension;
    every attaching map has contractible target and the space is a wedge.
    """
    if not dong:
        return WedgeOfSpheres.contractible()
    if any(not s for s in dong):
        return None
    dimensions = {len(s) - 1 for s in dong}
    if len(dimensions) != 1:
        return None
    dim = next(iter(dimensions))
    return WedgeOfSpheres({dim: len(dong)})


def _dong_matching_is_acyclic(
    s_complex: SimplicialComplex,
    critical: set[Simplex],
    pairs: list[tuple[Simplex, Simplex]],
) -> bool:
    """Independently validate matched Hasse edges and gradient acyclicity."""
    faces = s_complex.all_simplices()
    used: set[Simplex] = set()
    matched_edges: set[tuple[Simplex, Simplex]] = set()
    for low, high in pairs:
        if (
            low not in faces
            or high not in faces
            or len(high) != len(low) + 1
            or not low.issubset(high)
            or low in used
            or high in used
        ):
            return False
        used.update((low, high))
        matched_edges.add((low, high))
    if critical != faces - used:
        return False
    gradient = nx.DiGraph()
    gradient.add_nodes_from(faces)
    for high in faces:
        for vertex in high:
            low = Simplex(high - {vertex})
            if (low, high) in matched_edges:
                gradient.add_edge(low, high)
            else:
                gradient.add_edge(high, low)
    return bool(nx.is_directed_acyclic_graph(gradient))


# --- Vertex decomposability ---


def _is_shedding_vertex(s_complex: SimplicialComplex, vertex) -> bool:
    link = s_complex.link(vertex)
    deletion = s_complex.deletion(vertex)
    return len(deletion.facet_set & link.facet_set) == 0


def is_vertex_decomposable(s_complex: SimplicialComplex) -> bool:
    """Return ``True`` if *s_complex* is vertex decomposable."""
    if len(s_complex.facet_set) == 1:
        return True
    for v in s_complex.vertex_set:
        if (
            _is_shedding_vertex(s_complex, v)
            and is_vertex_decomposable(s_complex.link(v))
            and is_vertex_decomposable(s_complex.deletion(v))
        ):
            return True
    return False


def is_contractible_via_flag_apex(h: SimplicialComplex | nx.Graph) -> bool:
    """Return ``True`` if *h* is trivially contractible because it is a cone.

    A complex is a cone when some vertex (an "apex") belongs to every facet
    -- equivalently, for a :class:`networkx.Graph` interpreted as its clique
    complex, some vertex is adjacent to every other vertex (a universal
    vertex). This is a strictly weaker diagnostic than full contractibility:
    many complexes that arise as vertex links (e.g. in
    :func:`~pycombtop.simplex.link`) are contractible for more general
    reasons and will return ``False`` here even though they are, in fact,
    contractible. Useful to distinguish "trivially a cone" from
    "contractible via more general means" when inspecting links by hand.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.homotopy_type import is_contractible_via_flag_apex
    >>> is_contractible_via_flag_apex(nx.star_graph(3))
    True
    >>> is_contractible_via_flag_apex(nx.cycle_graph(5))
    False
    """
    if isinstance(h, nx.Graph):
        n = h.number_of_nodes()
        if n <= 1:
            return True
        return any(h.degree(v) == n - 1 for v in h.nodes())
    if not h.facet_set:
        return True
    return any(all(v in f for f in h.facet_set) for v in h.vertex_set)


# ---------------------------------------------------------------------------
# Individual strategies - each returns HomotopyVerdict | None
# ---------------------------------------------------------------------------


def _try_trivial(graph: nx.Graph) -> HomotopyVerdict | None:
    """One-vertex graph is immediately contractible."""
    if graph.order() == 1:
        return HomotopyVerdict(
            wedge=WedgeOfSpheres.contractible(),
            reason=Theorem.CONTRACTIBLE_POINT,
        )
    return None


def _try_dong(
    s_complex: SimplicialComplex, extra_reasons: str = ""
) -> HomotopyVerdict | None:
    """Try Dong's matching, accepting only verified acyclic matchings."""
    collapsed = collapse(s_complex)
    orderings: list[Callable[[set], list]] = [list] + [_shuff] * 6
    for ordering in orderings:
        critical, pairs = cast(
            tuple[set[Simplex], list[tuple[Simplex, Simplex]]],
            dong_matching(collapsed, order_function=ordering, return_pairs=True),
        )
        if not _dong_matching_is_acyclic(collapsed, critical, pairs):
            continue
        wedge = _read_dong(critical)
        if wedge is None:
            continue
        reason = Theorem.DONG_MATCHING
        if extra_reasons:
            reason = extra_reasons + " Then: " + reason
        return HomotopyVerdict(wedge=wedge, reason=reason)
    return None


def _try_vertex_decomposable(
    s_complex: SimplicialComplex,
    extra_reasons: str = "",
) -> HomotopyVerdict | None:
    """Try vertex decomposability to justify Betti-number-based homotopy type."""
    collapsed = collapse(s_complex)
    if is_vertex_decomposable(collapsed):
        bettis = betti_numbers_sc(collapsed)
        w = WedgeOfSpheres.from_betti(bettis)
        reason = Theorem.VERTEX_DECOMPOSABLE
        if extra_reasons:
            reason = extra_reasons + "  Then: " + reason
        return HomotopyVerdict(wedge=w, reason=reason)
    return None


def _try_join_complement(
    graph: nx.Graph,
    max_vertices: int | None = None,
    max_simplices: int | None = None,
) -> HomotopyVerdict | None:
    """Use the join formula only with certified homotopy types of factors."""
    complement = nx.complement(graph)
    components = list(nx.connected_components(complement))
    if len(components) < 2:
        return None
    factors = []
    for component in components:
        factor = nx.complement(complement.subgraph(component).copy())
        verdict = homotopy_type_with_verdict(
            factor, max_vertices=max_vertices, max_simplices=max_simplices
        )
        if not verdict.is_exact or verdict.is_limited:
            return None
        factors.append(verdict.wedge)
    # Reduced homology of a join is the shifted tensor product;
    # for certified wedges this gives all sphere multiplicities.
    spheres = dict(factors[0].spheres)
    for factor in factors[1:]:
        next_spheres: dict[int, int] = {}
        for d, n in spheres.items():
            for e, m in factor.spheres.items():
                dim = d + e + 1
                next_spheres[dim] = next_spheres.get(dim, 0) + n * m
        spheres = next_spheres
    return HomotopyVerdict(
        wedge=WedgeOfSpheres(spheres),
        reason=Theorem.JOIN_COMPLEMENT,
    )


def _try_universal_vertex_cone(graph: nx.Graph) -> HomotopyVerdict | None:
    """A universal vertex is an apex of the flag complex."""
    if graph.number_of_nodes() and any(
        degree == graph.number_of_nodes() - 1 for _, degree in graph.degree()
    ):
        return HomotopyVerdict(
            wedge=WedgeOfSpheres.contractible(),
            reason=Theorem.UNIVERSAL_VERTEX_CONE,
        )
    return None


def _is_complete(graph: nx.Graph) -> bool:
    n = graph.order()
    return bool(graph.size() == n * (n - 1) // 2)


def _complete_neighborhood_components(graph: nx.Graph, vertex) -> list[set] | None:
    """Return complete components of N(vertex), or None if inapplicable."""
    neighbors = set(graph.neighbors(vertex))
    if len(neighbors) < 2:
        return None
    neighborhood = graph.subgraph(neighbors)
    components = [set(c) for c in nx.connected_components(neighborhood)]
    if len(components) < 2:
        return None
    if not all(_is_complete(neighborhood.subgraph(c)) for c in components):
        return None
    return components


def _try_disjoint_complete_neighborhood(
    graph: nx.Graph,
) -> HomotopyVerdict | None:
    """Delete a vertex with r>=2 disjoint complete neighborhood components.

    The link in the flag complex consists of r contractible simplices.
    If deletion is connected, attaching the vertex star adds r-1 circles.
    """
    for vertex in graph:
        components = _complete_neighborhood_components(graph, vertex)
        if components is None:
            continue
        deletion = graph.copy()
        deletion.remove_node(vertex)
        if deletion.number_of_nodes() == 0 or not nx.is_connected(deletion):
            continue
        verdict = homotopy_type_with_verdict(deletion)
        if verdict.is_limited:
            return verdict
        return HomotopyVerdict(
            wedge=verdict.wedge.add_spheres(1, len(components) - 1),
            reason=Theorem.DISJOINT_COMPLETE_NEIGHBORHOOD
            + " Recursive result: "
            + verdict.reason,
            is_exact=verdict.is_exact,
        )
    return None


def homotopy_type_clique_graph_via_independent_neighborhood(
    graph: nx.Graph,
    vertex,
) -> HomotopyVerdict:
    """Classify Delta(K(graph)) from Delta(K(graph-vertex)).

    Requires r>=2 pairwise nonadjacent neighbors of vertex, plus connected
    deletion of order >=2. Otherwise raises ValueError rather than issuing
    an invalid homotopy certificate.

    All cliques of H=graph-vertex survive as cliques in graph, while each
    edge {vertex,neighbor} gives a new clique. These r new cliques form a
    simplex in Delta(K(graph)). Quotienting that simplex (a contractible
    subcomplex) to a point turns the remaining attachments into a cone
    over r disjoint simplices on the old cliques containing each neighbor.
    The connected deletion therefore contributes r-1 circles.
    """
    if vertex not in graph:
        raise ValueError("vertex is not in graph")
    neighborhood = list(graph.neighbors(vertex))
    if len(neighborhood) < 2:
        raise ValueError("vertex must have at least two neighbors")
    if graph.subgraph(neighborhood).number_of_edges() != 0:
        raise ValueError("vertex's open neighborhood must be independent")
    deletion = graph.copy()
    deletion.remove_node(vertex)
    if deletion.number_of_nodes() < 2 or not nx.is_connected(deletion):
        raise ValueError("graph minus vertex must be connected of order >= 2")
    clique_deletion = clique_graph(deletion)
    if clique_deletion is None:
        raise ValueError("could not construct the clique graph of deletion")
    verdict = homotopy_type_with_verdict(clique_deletion)
    if verdict.is_limited:
        return verdict
    return HomotopyVerdict(
        wedge=verdict.wedge.add_spheres(1, len(neighborhood) - 1),
        reason=Theorem.INDEPENDENT_NEIGHBORHOOD_CLIQUE_GRAPH
        + " Recursive result: "
        + verdict.reason,
        is_exact=verdict.is_exact,
    )


def _is_triangle_free_edge_candidate(graph: nx.Graph, edge) -> bool:
    n1 = set(open_neighborhood(graph, edge[0]))
    n2 = set(open_neighborhood(graph, edge[1]))
    return len(n1 & n2) == 0 and graph.degree(edge[0]) > 2 and graph.degree(edge[1]) > 2


def _try_triangle_free_intercomponent_edges(
    graph: nx.Graph,
) -> HomotopyVerdict | None:
    """Split two induced connected pieces joined by triangle-free edges.

    Every removed edge must run between the two pieces: removing an edge
    inside a piece would invalidate the asserted homotopy decomposition.
    """
    candidate_edges = [
        e for e in graph.edges() if _is_triangle_free_edge_candidate(graph, e)
    ]
    if not candidate_edges:
        return None
    reduced = graph.copy()
    reduced.remove_edges_from(candidate_edges)
    components = [set(c) for c in nx.connected_components(reduced)]
    if len(components) != 2:
        return None
    left, right = components
    if not all(
        (u in left and v in right) or (u in right and v in left)
        for u, v in candidate_edges
    ):
        return None
    # All cross edges must have been removed; each has no common neighbor.
    cross_edges = [
        (u, v)
        for u, v in graph.edges()
        if (u in left and v in right) or (u in right and v in left)
    ]
    if len(cross_edges) != len(candidate_edges):
        return None
    first = homotopy_type_with_verdict(graph.subgraph(left).copy())
    second = homotopy_type_with_verdict(graph.subgraph(right).copy())
    if first.is_limited or second.is_limited:
        return None
    if not first.is_exact or not second.is_exact:
        return None
    return HomotopyVerdict(
        wedge=first.wedge.wedge(second.wedge).add_spheres(1, len(candidate_edges) - 1),
        reason=Theorem.TRIANGLE_FREE_INTERCOMPONENT_EDGES,
    )


def _try_cutvertex_wedge(graph: nx.Graph) -> HomotopyVerdict | None:
    """Cutvertex pieces intersect in exactly one vertex, hence wedge."""
    if not nx.is_connected(graph):
        return None
    cutvertices = list(nx.articulation_points(graph))
    if not cutvertices:
        return None
    vertex = cutvertices[0]
    reduced = graph.copy()
    reduced.remove_node(vertex)
    components = list(nx.connected_components(reduced))
    pieces = [
        graph.subgraph(set(component) | {vertex}).copy() for component in components
    ]
    verdicts = [homotopy_type_with_verdict(piece) for piece in pieces]
    if any(not v.is_exact or v.is_limited for v in verdicts):
        return None
    wedge = WedgeOfSpheres.contractible()
    for v in verdicts:
        wedge = wedge.wedge(v.wedge)
    return HomotopyVerdict(wedge=wedge, reason=Theorem.CUTVERTEX_WEDGE)


def _try_contractible_link_components(
    s_complex: SimplicialComplex,
) -> HomotopyVerdict | None:
    """Handle a vertex with a disconnected, componentwise contractible link.

    Applies to arbitrary simplicial complexes, including non-flag ones.
    Each link component must have a *certified* contractible type and the
    deletion must have connected 1-skeleton. Under these exact hypotheses
    the cone on the link adds one circle per component beyond the first.
    """
    reduced = collapse(s_complex)
    for vertex in reduced.vertex_set:
        lk = reduced.link(vertex)
        skeleton = lk.one_skeleton_graph()
        components = [set(c) for c in nx.connected_components(skeleton)]
        if len(components) < 2:
            continue
        deletion = reduced.deletion(vertex)
        deletion_graph = deletion.one_skeleton_graph()
        if deletion_graph.number_of_nodes() == 0 or not nx.is_connected(deletion_graph):
            continue
        certified = True
        for component in components:
            subfacets = [f for f in lk.facet_set if f.issubset(component)]
            piece = SimplicialComplex(component, facet_set=subfacets)
            if len(piece.facet_set) == 1:
                continue  # One simplex is visibly a cone.
            check = homotopy_type_sc_with_verdict(piece)
            if not check.is_exact or not check.wedge.is_contractible():
                certified = False
                break
        if not certified:
            continue
        verdict = homotopy_type_sc_with_verdict(deletion)
        if verdict.is_limited:
            return verdict
        return HomotopyVerdict(
            wedge=verdict.wedge.add_spheres(1, len(components) - 1),
            reason=Theorem.CONTRACTIBLE_LINK_COMPONENTS
            + " Recursive result: "
            + verdict.reason,
            is_exact=verdict.is_exact,
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def homotopy_type_sc_with_verdict(
    s_complex: SimplicialComplex,
    max_vertices: int | None = None,
    max_simplices: int | None = None,
) -> HomotopyVerdict:
    """Compute the homotopy type of a simplicial complex *s_complex*.

    Tries strategies in order and returns the first :class:`HomotopyVerdict`
    that succeeds.  The *reason* field identifies the theorem used.

    .. rubric:: Parameters

    s_complex : SimplicialComplex
        Complex to analyze.
    max_vertices : int, optional
        Abort before analysis when the complex has more vertices than this.
    max_simplices : int, optional
        Abort before analysis when the complex has more distinct simplices
        than this. This is a size guard, not a timeout.

    Returns
    -------
    HomotopyVerdict
        *verdict* is a LaTeX string like ``"Contractible"``,
        ``"\\\\(S^{2}\\\\)"``, or ``"\\\\(\\\\vee_{3}S^{1}\\\\)"``.
        *is_exact* is ``False`` only when no topological shortcut was
        found and only Betti numbers are returned.
    """
    if _too_large_simplicial_complex(s_complex, max_vertices, max_simplices):
        return _limited_verdict()

    # 1. Dong matching on original complex
    v = _try_dong(s_complex)
    if v:
        return v

    # 2. Vertex with a disconnected union of contractible link components
    v = _try_contractible_link_components(s_complex)
    if v:
        return v

    # 3. Vertex decomposable
    v = _try_vertex_decomposable(s_complex)
    if v:
        return v

    # 4. Betti numbers only
    bettis = betti_numbers_sc(collapse(s_complex))
    return HomotopyVerdict(
        wedge=WedgeOfSpheres.from_betti(bettis),
        reason=Theorem.BETTI_ONLY,
        is_exact=False,
    )


def homotopy_type_with_verdict(
    graph: nx.Graph,
    max_vertices: int | None = None,
    max_simplices: int | None = None,
) -> HomotopyVerdict:
    """Compute the homotopy type of the clique complex of *graph*.

    .. rubric:: Parameters

    graph : networkx.Graph
        A NetworkX graph.
    max_vertices : int, optional
        Abort before analysis when the graph has more vertices than this.
    max_simplices : int, optional
        Maximum number of distinct faces for general simplicial-complex
        algorithms. Cheap exact cone/point/join certificates are tried first,
        without materializing the full complex. If none applies, a streaming
        face-count check stops at the configured budget.

    .. rubric:: Returns

    HomotopyVerdict
        A dataclass with fields:

        * ``verdict`` - LaTeX string for the homotopy type.
        * ``reason``  - The theorem(s) that justify the verdict.
        * ``is_exact`` - ``False`` when only Betti numbers were computed; in
          that case the returned sphere counts are Betti numbers, not a
          verified wedge-of-spheres decomposition.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.homotopy_type import homotopy_type_with_verdict
    >>> G = nx.cycle_graph(5)
    >>> v = homotopy_type_with_verdict(G)
    >>> v.verdict
    '\\\\(S^{1}\\\\)'
    """
    if max_vertices is not None and graph.number_of_nodes() > max_vertices:
        return _limited_verdict()
    if max_simplices is not None and max_simplices < 0:
        raise ValueError("max_simplices must be nonnegative or None")

    # Cheap exact certificates need no explicit enumeration of faces.
    # Preserve the strict vertex limit, but do not reject a cone or join
    # merely because the full flag complex is exponentially large.
    v = _try_trivial(graph)
    if v:
        return v
    v = _try_universal_vertex_cone(graph)
    if v:
        return v
    v = _try_join_complement(
        graph, max_vertices=max_vertices, max_simplices=max_simplices
    )
    if v:
        return v

    # Check the original complex budget BEFORE expensive s-collapses.
    # Streaming nx.find_cliques permits early termination even when there
    # would be exponentially many maximal cliques or distinct faces.
    if max_simplices is not None and _graph_exceeds_simplex_budget(
        graph, max_simplices
    ):
        return _limited_verdict()

    # 1. Simplify via s-collapses, then retry trivial
    graph = _simplify_graph_ht(graph)
    v = _try_trivial(graph)
    if v:
        # upgrade reason to mention the collapse
        return HomotopyVerdict(
            wedge=WedgeOfSpheres.contractible(),
            reason=Theorem.S_COLLAPSE_TO_POINT,
        )

    # 2. Universal vertex gives a cone (before the general join method)
    v = _try_universal_vertex_cone(graph)
    if v:
        return v

    # 3. Disconnected complement -> simplicial join
    v = _try_join_complement(graph)
    if v:
        return v

    # 4. Disjoint complete components in the open neighborhood
    v = _try_disjoint_complete_neighborhood(graph)
    if v:
        return v

    # 5. Triangle-free edges between two induced connected pieces
    v = _try_triangle_free_intercomponent_edges(graph)
    if v:
        return v

    # 6. Cutvertex wedge decomposition
    v = _try_cutvertex_wedge(graph)
    if v:
        return v

    # 7. Dong matching on clique complex (possibly after further simplification)
    sc = clique_complex(graph)
    v = _try_dong(sc)
    if v:
        return v

    graph2 = nx.convert_node_labels_to_integers(_simplify_graph_ht(graph))
    sc2 = clique_complex(graph2)
    v = _try_dong(sc2)
    if v:
        return v

    # 8. Disconnected contractible vertex link
    v = _try_contractible_link_components(sc2)
    if v:
        return v

    # 9. Vertex decomposable
    v = _try_vertex_decomposable(sc2)
    if v:
        return v

    # 10. Betti numbers only
    bettis = betti_numbers_sc(collapse(sc2))
    return HomotopyVerdict(
        wedge=WedgeOfSpheres.from_betti(bettis),
        reason=Theorem.BETTI_ONLY,
        is_exact=False,
    )


def homotopy_type_large_graph(graph: nx.Graph, bound: int = 100) -> HomotopyVerdict:
    """Compute reduced Betti numbers for graphs too large for combinatorial methods.

    .. rubric:: Parameters

    graph : networkx.Graph
        A NetworkX graph (typically with many vertices).
    bound : int, optional
        Vertex-count threshold below which
        :func:`homotopy_type_with_verdict` is called instead (default 100).

    .. rubric:: Returns

    HomotopyVerdict
        *is_exact* is always ``False``; the verdict is a Betti-number list.
    """
    if graph.order() < bound:
        return homotopy_type_with_verdict(graph)
    bettis = betti_numbers_graph(graph)
    return HomotopyVerdict(
        wedge=WedgeOfSpheres.from_betti(bettis),
        reason=Theorem.LARGE_GRAPH_BETTI,
        is_exact=False,
    )


# ---------------------------------------------------------------------------
# Star / star-cluster / intersection helpers (used by _try_star_cluster above
# and also useful as public utilities)
#
# `star` (used below) is re-exported here from `.simplex`, which is where
# the general (any-simplex, graph-or-complex) implementation lives, together
# with `link` and `deletion`.
# ---------------------------------------------------------------------------


def star_cluster(s_complex: SimplicialComplex, simplex) -> SimplicialComplex:
    """Return the star cluster of *simplex* (a set of vertices) in *s_complex*."""
    facets: set = set()
    for v in simplex:
        facets |= {f for f in s_complex.facet_set if v in f}
    vertices = set.union(*(set(s) for s in facets))
    return SimplicialComplex(vertices, facet_set=facets)


def intersection_complex(
    sc1: SimplicialComplex, sc2: SimplicialComplex
) -> SimplicialComplex:
    """Return the intersection of two simplicial complexes."""
    vertices = {x for x in sc1.vertex_set if sc2.function({x})}

    def _fn(s):
        return sc1.function(s) and sc2.function(s)

    return SimplicialComplex(vertices, function=_fn)
