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
.. [Dong2002] X. Dong, "The topology of bounded-degree graph complexes",
   J. Algebra 262 (2003) 287-312.

.. [BM1997] A. Björner & M. Wachs, "Shellable nonpure complexes and posets I",
   Trans. Amer. Math. Soc. 348 (1996) 1299-1327.  (Vertex decomposability.)

.. [Wh1939] J. H. C. Whitehead, "Simplicial spaces, nuclei and m-groups",
   Proc. London Math. Soc. 45 (1939) 243-327.  (Elementary collapses.)

.. [MV1930] K. Mayer-Vietoris.  (Nerve / Mayer-Vietoris sequence used in the
   join-complement, cutpoint, and special-edge theorems.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import mogutda
import networkx as nx
from pycliques.surfaces import open_neighborhood
from sympy import Add, Mul, Poly, symbols

from .s_collapses import complete_s_collapse, complete_s_collapse_edges
from .simplex import Simplex, SimplicialComplex, clique_complex

# ---------------------------------------------------------------------------
# Theorem citations
# ---------------------------------------------------------------------------


class Theorem:
    """Namespace of human-readable theorem labels used as ``reason`` values."""

    CONTRACTIBLE_POINT = "One-point complex is contractible (trivially)."
    COLLAPSIBLE = (
        "Complex is collapsible (sequence of elementary collapses); "
        "contractible by [Wh1939]."
    )
    DONG_MATCHING = (
        "Dong matching (acyclic matching on the Hasse diagram) yields a "
        "single critical cell in each dimension; homotopy type read off "
        "from the critical cells [Dong2002]."
    )
    VERTEX_DECOMPOSABLE = (
        "Complex (after collapse) is vertex decomposable [BM1997]; "
        "Betti numbers computed and homotopy type identified."
    )
    JOIN_COMPLEMENT = (
        "Complement graph is disconnected; clique complex is the join of "
        "the clique complexes of each complementary component. "
        "Join formula applied via Künneth / suspension [MV1930]."
    )
    STAR_CLUSTER = (
        "Isolated vertex in complement graph detected; star-cluster "
        "decomposition applied [MV1930]."
    )
    SPECIAL_NEIGHBOURHOOD = (
        "Vertex whose open neighbourhood is a disjoint union of two "
        "complete graphs detected; Mayer-Vietoris / pushout argument "
        "gives +1 to the S^1 count [MV1930]."
    )
    SPECIAL_EDGES = (
        "Bridge edges (no common neighbour, both endpoints have degree >= 3) "
        "detected; graph splits into components whose complexes are vertex "
        "decomposable; polynomial addition formula applied [MV1930]."
    )
    CUTPOINT = (
        "Cut-point detected; complex splits as a wedge by a "
        "Mayer-Vietoris argument; each piece is vertex decomposable "
        "[MV1930]."
    )
    SPECIAL_CUTPOINT_KG = (
        "Special cut-point in G used to compute homotopy type of K(G) via "
        "a pushout / Mayer-Vietoris argument on the clique graph [MV1930]."
    )
    SPECIAL_VERTEX_SC = (
        "Vertex with 0-dimensional link found in simplified complex; "
        "deletion / link decomposition applied recursively."
    )
    BETTI_ONLY = (
        "No combinatorial shortcut succeeded; Betti numbers computed "
        "directly (homotopy type may not be fully determined from Betti "
        "numbers alone)."
    )
    LARGE_GRAPH_BETTI = (
        "Graph too large for combinatorial homotopy-type methods; "
        "reduced Betti numbers computed via mogutda."
    )


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
        The structured homotopy type when known to be a wedge of spheres.
        ``None`` when only Betti numbers could be computed.
    reason : str
        The theorem or sequence of reductions used to establish the
        verdict.  One of the :class:`Theorem` constants, or a
        concatenation of several.
    is_exact : bool
        ``True`` when the verdict is provably the correct homotopy type.
        ``False`` when only Betti numbers could be computed.
    """

    wedge: WedgeOfSpheres
    reason: str
    is_exact: bool = True

    @property
    def verdict(self) -> str:
        """A human-readable (LaTeX-compatible) string for the homotopy type."""
        return self.wedge.to_latex()

    def __str__(self) -> str:  # pragma: no cover
        exact_tag = "" if self.is_exact else " [Betti numbers only]"
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


# --- Simplicial-complex collapse (elementary collapses) ---


def _facets_containing(s_complex: SimplicialComplex, simplex: Simplex) -> int:
    return sum(1 for f in s_complex.facet_set if simplex.issubset(f))


def _is_free_face(s_complex: SimplicialComplex, simplex: Simplex) -> bool:
    return (
        simplex not in s_complex.facet_set
        and _facets_containing(s_complex, simplex) == 1
    )


def _remove_simplex(
    s_complex: SimplicialComplex, simplex: Simplex
) -> SimplicialComplex:
    facet = next(f for f in s_complex.facet_set if simplex.issubset(f))
    others = s_complex.facet_set - {facet}
    good = others.copy()
    for v in simplex:
        reduced = Simplex(facet - {v})
        if not any(reduced.issubset(f) for f in others):
            good = good | {reduced}
    vertices = set.union(*(set(s) for s in good))
    return SimplicialComplex(vertices, facet_set=good)


def _has_free_face(s_complex: SimplicialComplex) -> Simplex | None:
    for face in s_complex.all_simplices():
        if _is_free_face(s_complex, face):
            return face
    return None


def collapse(s_complex: SimplicialComplex) -> SimplicialComplex:
    """Collapse *s_complex* by repeated elementary collapses."""
    sc = s_complex
    while True:
        if len(sc.vertex_set) in {0, 1}:
            return sc
        free_face = _has_free_face(sc)
        if free_face is None:
            return sc
        if len(free_face) == 0:
            v = next(iter(sc.vertex_set))
            return SimplicialComplex({v}, facet_set={Simplex({v})})
        sc = _remove_simplex(sc, free_face)


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
    numbers = [mc.betti_number(i) for i in range(dim + 1)]
    numbers[0] -= 1  # reduced beta_0
    return _simplify_betti_list(numbers)


def betti_numbers_graph(graph: nx.Graph) -> list[int]:
    """Reduced Betti numbers of the clique complex of *graph* via mogutda."""
    simplices = [tuple(c) for c in nx.find_cliques(graph)]
    mc = mogutda.SimplicialComplex(simplices=simplices)
    dim = max((len(s) - 1 for s in simplices), default=-1)
    numbers = [mc.betti_number(i) for i in range(dim + 1)]
    numbers[0] -= 1
    return _simplify_betti_list(numbers)


def _read_dong(dong) -> WedgeOfSpheres | None:
    """Interpret a Dong matching into a WedgeOfSpheres, or None if ambiguous."""
    n = len(dong)
    if n == 0:
        return WedgeOfSpheres.contractible()
    lst = list(dong)
    dim = len(lst[0])
    if n == 1:
        return WedgeOfSpheres.sphere(dim - 1)
    if all(len(s) == dim for s in lst):
        return WedgeOfSpheres({dim - 1: n})
    return None


# --- Polynomial helpers for Mayer-Vietoris arguments ---


def _list_to_poly(lst: list[int]) -> Poly:
    x = symbols("x")
    return Poly(sum(c * x**p for p, c in enumerate(lst)), x)


def _poly_to_list(poly: Poly) -> list[int]:
    coeffs: list[int] = list(poly.all_coeffs())
    coeffs.reverse()
    return coeffs


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
    """Try Dong matching (up to 7 attempts with random vertex orderings)."""
    collapsed = collapse(s_complex)
    # deterministic first pass
    w = _read_dong(collapsed.dong_matching())
    if w is not None:
        reason = Theorem.DONG_MATCHING
        if extra_reasons:
            reason = extra_reasons + "  Then: " + reason
        return HomotopyVerdict(wedge=w, reason=reason)
    # randomised passes
    for _ in range(6):
        w = _read_dong(collapsed.dong_matching(order_function=_shuff))
        if w is not None:
            reason = Theorem.DONG_MATCHING
            if extra_reasons:
                reason = extra_reasons + "  Then: " + reason
            return HomotopyVerdict(wedge=w, reason=reason)
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


def _try_join_complement(graph: nx.Graph) -> HomotopyVerdict | None:
    """Disconnected complement -> clique complex is a join."""
    x = symbols("x")
    cg = nx.complement(nx.convert_node_labels_to_integers(graph))
    comps = [cg.subgraph(c).copy() for c in nx.connected_components(cg)]
    if len(comps) <= 1:
        return None
    sub_complexes = [clique_complex(nx.complement(s)) for s in comps]
    collapsed = [collapse(sc) for sc in sub_complexes]
    if not all(is_vertex_decomposable(c) for c in collapsed):
        return None
    polys = [_list_to_poly(betti_numbers_sc(c)) for c in collapsed]
    product = Poly(Mul(*[p.as_expr() for p in polys]), x)
    padding = [0] * (len(comps) - 1)
    bettis = padding + _poly_to_list(product)
    return HomotopyVerdict(
        wedge=WedgeOfSpheres.from_betti(bettis),
        reason=Theorem.JOIN_COMPLEMENT,
    )


def _try_star_cluster(graph: nx.Graph) -> HomotopyVerdict | None:
    """Isolated vertex in complement -> star-cluster decomposition."""
    if graph.order() == 1:
        return HomotopyVerdict(
            wedge=WedgeOfSpheres.contractible(),
            reason=Theorem.CONTRACTIBLE_POINT,
        )
    graph = nx.convert_node_labels_to_integers(graph)
    cg = nx.complement(graph)
    isolated = [v for v in cg.nodes() if open_neighborhood(cg, v).size() == 0]
    if not isolated:
        return None
    vertex = isolated[0]
    ig = clique_complex(graph)
    st = star(ig, vertex)
    sc = star_cluster(ig, cg[vertex])
    inter = collapse(intersection_complex(st, sc))
    # empty intersection means the decomposition is not applicable
    if not inter.vertex_set:
        return None
    # check vertex decomposability of intersection
    if not is_vertex_decomposable(inter):
        return None
    bettis = [0] + betti_numbers_sc(inter)
    return HomotopyVerdict(
        wedge=WedgeOfSpheres.from_betti(bettis),
        reason=Theorem.STAR_CLUSTER,
    )


def _is_complete(graph: nx.Graph) -> bool:
    n = graph.order()
    return bool(graph.size() == n * (n - 1) // 2)


def _is_disjoint_union_of_completes(graph: nx.Graph) -> bool:
    comps = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    return len(comps) == 2 and all(_is_complete(h) for h in comps)


def _try_special_neighbourhood(graph: nx.Graph) -> HomotopyVerdict | None:
    """Vertex with open neighbourhood = K_p disjoint union of K_q -> +1 to S^1 count."""
    neighs = [(v, open_neighborhood(graph, v)) for v in graph.nodes()]
    candidates = [
        v
        for (v, nei) in neighs
        if _is_disjoint_union_of_completes(nei)
        and nx.is_connected(graph.subgraph(set(graph.nodes()) - {v}))
    ]
    if not candidates:
        return None
    v = candidates[0]
    sub = nx.convert_node_labels_to_integers(graph.subgraph(set(graph.nodes()) - {v}))
    inner = homotopy_type_with_verdict(sub)
    w = inner.wedge.add_spheres(1, 1)
    return HomotopyVerdict(
        wedge=w,
        reason=Theorem.SPECIAL_NEIGHBOURHOOD,
    )


def _is_special_edge(graph: nx.Graph, edge) -> bool:
    n1 = set(open_neighborhood(graph, edge[0]))
    n2 = set(open_neighborhood(graph, edge[1]))
    return len(n1 & n2) == 0 and graph.degree(edge[0]) > 2 and graph.degree(edge[1]) > 2


def _try_special_edges(graph: nx.Graph) -> HomotopyVerdict | None:
    """Bridge-like edges with no common neighbour -> polynomial addition."""
    x = symbols("x")
    graph = _simplify_graph_ht(graph)
    if not nx.is_connected(graph):
        return None
    sp_edges = [e for e in graph.edges() if _is_special_edge(graph, e)]
    if not sp_edges:
        return None
    cg = graph.copy()
    cg.remove_edges_from(sp_edges)
    comps = [cg.subgraph(c).copy() for c in nx.connected_components(cg)]
    if len(comps) != 2:
        return None
    sub_complexes = [clique_complex(s) for s in comps]
    collapsed = [collapse(sc) for sc in sub_complexes]
    if not all(is_vertex_decomposable(c) for c in collapsed):
        return None
    polys = [_list_to_poly(betti_numbers_sc(c)) for c in collapsed]
    polys.append(_list_to_poly([0, len(sp_edges) - 1]))
    total = Poly(Add(*[p.as_expr() for p in polys]), x)
    bettis = _poly_to_list(total)
    return HomotopyVerdict(
        wedge=WedgeOfSpheres.from_betti(bettis),
        reason=Theorem.SPECIAL_EDGES,
    )


def _try_cutpoints(graph: nx.Graph) -> HomotopyVerdict | None:
    """Cut-point -> Mayer-Vietoris wedge."""
    x = symbols("x")
    graph = _simplify_graph_ht(graph)
    if not nx.is_connected(graph):
        return None
    cutpoints = list(nx.articulation_points(graph))
    if not cutpoints:
        return None
    vertex = cutpoints[0]
    cg = graph.copy()
    cg.remove_node(vertex)
    pieces = [graph.subgraph(comp | {vertex}) for comp in nx.connected_components(cg)]
    sub_complexes = [clique_complex(g) for g in pieces]
    collapsed = [collapse(sc) for sc in sub_complexes]
    if not all(is_vertex_decomposable(c) for c in collapsed):
        return None
    polys = [_list_to_poly(betti_numbers_sc(c)) for c in collapsed]
    total = Poly(Add(*[p.as_expr() for p in polys]), x)
    bettis = _poly_to_list(total)
    return HomotopyVerdict(
        wedge=WedgeOfSpheres.from_betti(bettis),
        reason=Theorem.CUTPOINT,
    )


def _try_special_vertex_sc(
    s_complex: SimplicialComplex,
) -> HomotopyVerdict | None:
    """Vertex with 0-dimensional link in collapsed complex."""
    sc = collapse(s_complex)
    for vertex in sc.vertex_set:
        if sc.link(vertex).dimension() == 0:
            deletion = sc.deletion(vertex)
            inner = homotopy_type_sc_with_verdict(deletion)
            n_neigh = len(sc.link(vertex).vertex_set)
            w = inner.wedge.add_spheres(1, n_neigh - 1)
            return HomotopyVerdict(
                wedge=w,
                reason=Theorem.SPECIAL_VERTEX_SC,
            )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def homotopy_type_sc_with_verdict(
    s_complex: SimplicialComplex,
) -> HomotopyVerdict:
    """Compute the homotopy type of a simplicial complex *s_complex*.

    Tries strategies in order and returns the first :class:`HomotopyVerdict`
    that succeeds.  The *reason* field identifies the theorem used.

    Parameters
    ----------
    s_complex:
        A :class:`~pycliques.simplicial.SimplicialComplex`.

    Returns
    -------
    HomotopyVerdict
        *verdict* is a LaTeX string like ``"Contractible"``,
        ``"\\\\(S^{2}\\\\)"``, or ``"\\\\(\\\\vee_{3}S^{1}\\\\)"``.
        *is_exact* is ``False`` only when no topological shortcut was
        found and only Betti numbers are returned.
    """
    # 1. Dong matching on original complex
    v = _try_dong(s_complex)
    if v:
        return v

    # 2. Special vertex with 0-dimensional link
    v = _try_special_vertex_sc(s_complex)
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


def homotopy_type_with_verdict(graph: nx.Graph) -> HomotopyVerdict:
    """Compute the homotopy type of the clique complex of *graph*.

    .. rubric:: Parameters

    graph : networkx.Graph
        A NetworkX graph.

    .. rubric:: Returns

    HomotopyVerdict
        A dataclass with fields:

        * ``verdict`` - LaTeX string for the homotopy type.
        * ``reason``  - The theorem(s) that justify the verdict.
        * ``is_exact`` - ``False`` when only Betti numbers were computed.

    .. rubric:: Examples
    >>> import networkx as nx
    >>> from pycombtop.homotopy_type import homotopy_type_with_verdict
    >>> G = nx.cycle_graph(5)
    >>> v = homotopy_type_with_verdict(G)
    >>> v.verdict
    '\\\\(S^{1}\\\\)'
    """
    # 0. Trivial case
    v = _try_trivial(graph)
    if v:
        return v

    # 1. Simplify via s-collapses, then retry trivial
    graph = _simplify_graph_ht(graph)
    v = _try_trivial(graph)
    if v:
        # upgrade reason to mention the collapse
        return HomotopyVerdict(
            wedge=WedgeOfSpheres.contractible(),
            reason=Theorem.COLLAPSIBLE,
        )

    # 2. Disconnected complement -> join
    v = _try_join_complement(graph)
    if v:
        return v

    # 3. Star-cluster (isolated vertex in complement)
    v = _try_star_cluster(graph)
    if v:
        return v

    # 4. Special neighbourhood
    v = _try_special_neighbourhood(graph)
    if v:
        return v

    # 5. Special bridge-like edges
    v = _try_special_edges(graph)
    if v:
        return v

    # 6. Cut-points
    v = _try_cutpoints(graph)
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

    # 8. Special vertex in the complex
    v = _try_special_vertex_sc(sc2)
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
# ---------------------------------------------------------------------------


def star(s_complex: SimplicialComplex, vertex) -> SimplicialComplex:
    """Return the star of *vertex* in *s_complex*."""
    facets = {f for f in s_complex.facet_set if vertex in f}
    vertices = set.union(*(set(s) for s in facets))
    return SimplicialComplex(vertices, facet_set=facets)


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
