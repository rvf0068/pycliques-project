from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from itertools import chain, combinations
from typing import Literal

import networkx as nx
from networkx.algorithms import tournament
from sympy import SparseMatrix


class Simplex(frozenset):
    """An immutable set of vertices representing a simplex.

    This class derives from :class:`frozenset` but overrides ``__repr__`` so
    that instances display like plain set literals instead of ``frozenset``.

    .. rubric:: Examples

    >>> from pycombtop import Simplex
    >>> s = Simplex({1, 2, 3})
    >>> s.dimension()
    2
    >>> Simplex([])
    {}
    """

    def __new__(cls, elements: Iterable) -> Simplex:
        return super().__new__(cls, elements)

    def __repr__(self) -> str:
        u = set(self)
        if len(u) == 0:
            return "{}"
        return f"{u}"

    def dimension(self) -> int:
        """Return the dimension of the simplex.

        The dimension equals the number of vertices minus one.

        .. rubric:: Examples

        >>> from pycombtop import Simplex
        >>> Simplex({0, 1, 2}).dimension()
        2
        >>> Simplex({5}).dimension()
        0
        >>> Simplex([]).dimension()
        -1
        """
        return len(self) - 1


class SimplicialComplex:
    """A simplicial complex over a finite base set.

    A :class:`SimplicialComplex` is composed of a base set of points and a set
    of simplices (of type :class:`Simplex`), which correspond to subsets of the
    base set.  The *vertex set* is the subset of the base set whose singletons
    are simplices.  It can be constructed either from an explicit set of facets
    or from a membership function that decides which subsets are simplices.

    .. rubric:: Parameters

    base_set : set
        The ground set of points.
    facet_set : set of sets, optional
        Maximal simplices.  When given, the membership function is derived
        automatically.
    function : callable, optional
        A predicate ``f(s) -> bool`` that returns ``True`` when a subset ``s``
        belongs to the complex.  Ignored when *facet_set* is provided.

    .. rubric:: Examples

    Build a complex from explicit facets::

        >>> from pycombtop import Simplex, SimplicialComplex
        >>> sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
        >>> sc.dimension()
        2

    Build a complex using a membership function::

        >>> sc2 = SimplicialComplex({0, 1, 2}, function=lambda s: len(s) <= 2)
        >>> sc2.dimension()
        1
    """

    def __init__(
        self,
        base_set: Iterable,
        facet_set: Iterable | None = None,
        function: Callable[..., bool] | None = None,
    ) -> None:
        self.base_set = set(base_set)

        self.function: Callable[..., bool]
        if facet_set is None:
            self.function = (
                function if function is not None else self._default_is_simplex
            )
            self.facet_set = self._facet_set_from_function()
        else:
            self.function = self._default_is_simplex
            faces = {Simplex(s) for s in facet_set}
            if any(not face <= self.base_set for face in faces):
                raise ValueError("generating face outside the base set")
            self.facet_set = self._maximal_faces(faces)

    @staticmethod
    def _maximal_faces(faces: set[Simplex]) -> set[Simplex]:
        """Index larger facets by vertex to discard redundant generators."""
        if len({len(face) for face in faces}) <= 1:
            return faces
        facets: set[Simplex] = set()
        containing: dict[object, list[Simplex]] = defaultdict(list)
        for face in sorted(faces, key=len, reverse=True):
            if not face:
                continue  # A larger generator is present.
            candidates = min((containing[v] for v in face), key=len)
            if not any(face < other for other in candidates):
                facets.add(face)
                for vertex in face:
                    containing[vertex].append(face)
        return facets

    @classmethod
    def _from_maximal_faces(
        cls, base_set: Iterable, facets: Iterable
    ) -> SimplicialComplex:
        """Construct from a family already known to be inclusion-maximal."""
        result = cls.__new__(cls)
        result.base_set = set(base_set)
        result.facet_set = {Simplex(face) for face in facets}
        result.function = result._default_is_simplex
        return result

    def _default_is_simplex(self, s: set) -> bool:
        """Fallback membership function if none is provided."""
        if not hasattr(self, "facet_set") or self.facet_set is None:  # pragma: no cover
            return False
        return any(s <= facet for facet in self.facet_set)

    def _facet_set_from_function(self) -> set[Simplex]:
        """Derive facets using the membership function.

        This calculates the facet set by finding the maximal subsets
        of vertices that satisfy *self.function*.
        """
        if self.function(self.base_set):
            return {Simplex(self.base_set)}

        facets: set[Simplex] = set()
        # Evaluate subsets lazily to save memory
        for r in reversed(range(len(self.base_set))):
            for subset in combinations(self.base_set, r):
                s = Simplex(subset)
                if self.function(s):
                    # Check if s is already subsumed by a known facet
                    if not any(s.issubset(f) for f in facets):
                        facets.add(s)
        return facets

    def __repr__(self) -> str:
        base = f"Simplicial complex with base_set {self.base_set} "
        return base + f"and facets {self.facet_set}."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimplicialComplex):
            return False

        return self.base_set == other.base_set and self.facet_set == other.facet_set

    def dimension(self) -> int:
        """Return the dimension of the simplicial complex.

        The dimension is the maximum dimension of its facets, or ``-1`` when
        the complex has no facets.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
        >>> sc.dimension()
        2
        >>> SimplicialComplex(set(), facet_set=set()).dimension()
        -1
        """
        if not self.facet_set:
            return -1
        return max(facet.dimension() for facet in self.facet_set)

    def deletion(self, x) -> SimplicialComplex:
        """Return the complex obtained by removing vertex *x*.

        All facets containing *x* are replaced by their faces that do not
        contain *x*, keeping only the maximal ones.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
        >>> d = sc.deletion(2)
        >>> d.dimension()
        1
        >>> 2 in d.base_set
        False
        """
        vertices = self.base_set - {x}
        return SimplicialComplex(vertices, facet_set=(f - {x} for f in self.facet_set))

    def link(self, x) -> SimplicialComplex:
        """Return the link of vertex *x* in the complex.

        The link consists of all simplices *s* such that *s* does not contain
        *x* but s union {x} is a simplex of the complex.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
        >>> lk = sc.link(0)
        >>> lk.dimension()
        1
        >>> 0 in lk.base_set
        False
        """
        containing = {f for f in self.facet_set if x in f}
        new_facets = {f - {x} for f in containing}

        if not new_facets:
            return SimplicialComplex(set(), facet_set=set())

        vertices = set.union(*(set(s) for s in new_facets))
        return SimplicialComplex(vertices, facet_set=new_facets)

    def skeleton(self, n: int) -> SimplicialComplex:
        """Return the *n*-skeleton of the complex.

        The *n*-skeleton keeps only simplices of dimension at most *n*.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
        >>> sk = sc.skeleton(1)
        >>> sk.dimension()
        1
        """

        facets = {f for f in self.facet_set if f.dimension() < n}
        facets.update(self.faces_of_dimension(n))
        return self._from_maximal_faces(self.base_set, facets)

    def faces_of_dimension(self, dimension: int) -> Generator[Simplex]:
        """Yield the distinct faces of a single dimension, including -1."""
        if dimension < -1:
            return
        seen: set[Simplex] = set()
        for facet in self.facet_set:
            for subset in combinations(facet, dimension + 1):
                face = Simplex(subset)
                if face not in seen:
                    seen.add(face)
                    yield face

    @property
    def vertex_set(self) -> set:
        """Return the set of actual vertices (0-simplices) of the complex.

        A point *x* in the base set is a vertex when ``{x}`` is a simplex.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}])
        >>> sc.vertex_set == {0, 1}
        True
        >>> sc.base_set == {0, 1, 2}
        True
        """
        return set().union(*self.facet_set)

    def one_skeleton_graph(self) -> nx.Graph:
        """Return the 1-skeleton as a :class:`networkx.Graph`.

        Nodes are the actual vertices (0-simplices) and edges are the
        1-simplices of the complex.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
        >>> g = sc.one_skeleton_graph()
        >>> sorted(g.nodes())
        [0, 1, 2]
        >>> g.number_of_edges()
        3
        """
        the_graph = nx.Graph()
        the_graph.add_nodes_from(self.vertex_set)
        the_graph.add_edges_from(tuple(face) for face in self.faces_of_dimension(1))
        return the_graph

    def is_clique_complex(self) -> bool:
        """Return whether this complex is a clique complex.

        This checks whether the complex equals the clique complex of
        its 1-skeleton.

        .. rubric:: Examples

        >>> import networkx as nx
        >>> from pycombtop import clique_complex
        >>> cc = clique_complex(nx.cycle_graph(5))
        >>> cc.is_clique_complex()
        True
        >>> sc = SimplicialComplex({0, 1, 2},
        ...     facet_set=[{0, 1}, {1, 2}, {0, 2}])
        >>> sc.is_clique_complex()
        False
        """
        return self.facet_set == clique_complex(self.one_skeleton_graph()).facet_set

    def all_simplices(self) -> set[Simplex]:
        """Return the set of all simplices in the complex.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
        >>> sorted(len(s) for s in sc.all_simplices())
        [0, 1, 1, 2]
        """
        all_simplices_set: set[tuple] = set()
        for facet in self.facet_set:
            s = list(facet)
            # chain.from_iterable creates a generator, set() consumes it.
            subset_generator = chain.from_iterable(
                combinations(s, r) for r in range(len(s) + 1)
            )
            all_simplices_set.update(subset_generator)
        return {Simplex(s) for s in all_simplices_set}


# --- Module Functions ---


def _stable_vertex_key(vertex):
    """Deterministic key for standard hashable vertex labels, including cliques."""
    if isinstance(vertex, (set, frozenset)):
        return ("set", tuple(sorted(_stable_vertex_key(v) for v in vertex)))
    if isinstance(vertex, tuple):
        return ("tuple", tuple(_stable_vertex_key(v) for v in vertex))
    return ("atom", type(vertex).__name__, repr(vertex))


def _stable_vertex_order(vertices: set) -> list:
    """Avoid the process-dependent iteration order of a set."""
    return sorted(vertices, key=_stable_vertex_key)


def dong_matching(
    s_complex: SimplicialComplex,
    order_function: Callable[[set], list] = _stable_vertex_order,
    *,
    return_pairs: bool = False,
) -> set[Simplex] | tuple[set[Simplex], list[tuple[Simplex, Simplex]]]:
    """Compute Dong's vertex-ordered discrete Morse matching.

    At each vertex, pair as-yet-unmatched faces in its link with their
    union with that vertex. The empty face is included: this is the
    augmented/reduced matching, whose first pair supplies the basepoint.
    Dong, *Topology of bounded-degree graph complexes*, J. Algebra 262
    (2003), 287--312, uses Forman's discrete Morse method (Forman,
    *Morse theory for cell complexes*, Adv. Math. 134 (1998), 90--145).

    Return unmatched faces, or (unmatched faces, matched codimension-one
    pairs) when return_pairs=True. The latter permits independent
    verification of acyclicity before certifying a homotopy type.

    .. rubric:: Parameters

    s_complex : :class:`SimplicialComplex`
        The complex on which to run the matching.
    order_function : callable, optional
        A function that converts the vertex set into an ordered list
        (default: stable order by vertex label).

    .. rubric:: Returns

    set of :class:`Simplex`
        The unmatched (critical) simplices.

    .. rubric:: Examples

    >>> from pycombtop import SimplicialComplex, dong_matching
    >>> sc = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    >>> critical = dong_matching(sc, order_function=sorted)
    >>> len(critical) % 2  # critical simplices are not paired
    0
    """
    matched: set[Simplex] = set()
    pairs: list[tuple[Simplex, Simplex]] = []
    vertices = order_function(s_complex.vertex_set)
    # Face order matters: an early pair can exclude a later one. Sort by
    # stable vertex ranks rather than the iteration order of a Python set.
    rank = {v: i for i, v in enumerate(_stable_vertex_order(s_complex.vertex_set))}

    for vertex in vertices:
        the_link = s_complex.link(vertex)
        link_simplices = the_link.all_simplices()

        for s in sorted(
            link_simplices,
            key=lambda face: (len(face), tuple(sorted(rank[v] for v in face))),
        ):
            s_plus_v = Simplex(s | {vertex})
            if (s not in matched) and (s_plus_v not in matched):
                matched.add(s)
                matched.add(s_plus_v)
                pairs.append((s, s_plus_v))

    critical = s_complex.all_simplices() - matched
    return (critical, pairs) if return_pairs else critical


def all_subsets(the_set: set) -> Generator[Simplex]:
    """Yield all non-empty subsets of *the_set* as :class:`Simplex` objects.

    Subsets are yielded in decreasing order of cardinality.

    .. rubric:: Examples

    >>> from pycombtop import all_subsets
    >>> subs = list(all_subsets({0, 1}))
    >>> len(subs)
    3
    >>> all(isinstance(s, Simplex) for s in subs)
    True
    """
    n = len(the_set)
    subsets = chain.from_iterable(
        combinations(the_set, r) for r in reversed(range(1, n + 1))
    )
    for x in subsets:
        yield Simplex(x)


def relative_betti_numbers(
    big_complex_maximal_faces: Iterable[Iterable],
    small_complex_membership_test: Callable[[Iterable], bool],
    *,
    method: Literal["exact", "approximate"] = "exact",
) -> dict[int, int]:
    """Return relative Betti numbers by exact rational or approximate rank.

    The ambient complex is given by its maximal faces, and the subcomplex is
    described by a predicate ``small_complex_membership_test(sigma)``.  The
    relative chain groups are generated from simplices of the ambient complex
    that are not in the subcomplex; boundary terms that land in the subcomplex
    are projected to zero.

    .. rubric:: Parameters

    big_complex_maximal_faces : iterable
        Iterable of maximal simplices of the ambient complex, each a set-like
        object such as a set, tuple, or frozenset.
    small_complex_membership_test : callable
        Predicate that returns ``True`` exactly when a simplex lies in the
        subcomplex.
    method : {"exact", "approximate"}
        ``"exact"`` (the default) computes boundary ranks over :math:`Q` using
        exact integer matrices. ``"approximate"`` uses floating-point
        ``numpy.linalg.matrix_rank`` as a fast screening mode.

    Notes
    -----
    Rational Betti numbers do not detect torsion in integral homology, and
    vanishing relative Betti numbers do not imply that the pair collapses.

    .. rubric:: Returns

    dict[int, int]
        A dictionary mapping each nonnegative degree *d* to the *d*-th Betti
        number of the relative complex ``(big, small)``.
    """
    if method not in {"exact", "approximate"}:
        raise ValueError("method must be 'exact' or 'approximate'")

    facets = [Simplex(f) for f in big_complex_maximal_faces]
    if not facets:
        return {}

    base_set = set().union(*(set(f) for f in facets))
    big_complex = SimplicialComplex(base_set, facet_set=facets)
    all_simplices = big_complex.all_simplices()
    vertex_order = {
        vertex: index for index, vertex in enumerate(sorted(base_set, key=repr))
    }

    by_dim: dict[int, list[Simplex]] = defaultdict(list)
    for sigma in all_simplices:
        if sigma.dimension() < 0 or small_complex_membership_test(sigma):
            continue
        by_dim[sigma.dimension()].append(sigma)

    if not by_dim:
        return {}

    max_dim = max(by_dim)

    def boundary_rank(d: int) -> int:
        cols, rows = by_dim.get(d, []), by_dim.get(d - 1, [])
        if not cols or not rows:
            return 0
        positions: dict[frozenset, int] = {face: i for i, face in enumerate(rows)}
        entries = {}
        for j, simplex in enumerate(cols):
            ordered = sorted(simplex, key=vertex_order.__getitem__)
            for k, vertex in enumerate(ordered):
                i = positions.get(simplex - {vertex})
                if i is not None:
                    entries[i, j] = (-1) ** k
        if method == "exact":
            # The domain matrix performs exact elimination over the integers.
            return int(SparseMatrix(len(rows), len(cols), entries).to_DM().rank())
        import numpy as np

        matrix = np.zeros((len(rows), len(cols)), dtype=int)
        for (i, j), value in entries.items():
            matrix[i, j] = value
        return int(np.linalg.matrix_rank(matrix))

    ranks = [boundary_rank(d) for d in range(max_dim + 2)]
    return {d: len(faces) - ranks[d] - ranks[d + 1] for d, faces in by_dim.items()}


def nerve_of_sets(sets: Iterable[set | frozenset]) -> SimplicialComplex:
    """Return the nerve of a collection of sets.

    The nerve is the simplicial complex whose vertices are the given sets and
    whose simplices are sub-collections with non-empty common intersection.

    .. rubric:: Examples

    >>> from pycombtop import nerve_of_sets
    >>> n = nerve_of_sets([{1, 2}, {2, 3}, {3, 4}])
    >>> n.dimension()
    1
    >>> n2 = nerve_of_sets([{1, 2}, {2, 3}, {1, 2, 3}])
    >>> n2.dimension()
    2
    """

    # Every simplex of the nerve is contained in the incidence star of a
    # common element. Its maximal faces are precisely the maximal such stars.
    vertices = {Simplex(s) for s in sets}
    incidence: dict = defaultdict(set)
    for member in vertices:
        for element in member:
            incidence[element].add(member)
    maximal: list[Simplex] = []
    for star in sorted({Simplex(s) for s in incidence.values()}, key=len, reverse=True):
        if not any(star <= facet for facet in maximal):
            maximal.append(star)
    return SimplicialComplex(vertices, facet_set=maximal)


def delta_of_clique_set(clique_set: Iterable[frozenset]) -> SimplicialComplex:
    """Union of the full simplices generated by the supplied cliques."""
    faces = [Simplex(clique) for clique in clique_set]
    return SimplicialComplex(set().union(*faces), facet_set=faces)


def clique_complex(graph: nx.Graph) -> SimplicialComplex:
    """Return the clique complex of an undirected graph.

    The clique complex has the vertices of *graph* as its vertex set and the
    maximal cliques as facets.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import clique_complex
    >>> cc = clique_complex(nx.cycle_graph(5))
    >>> cc.dimension()
    1
    >>> cc = clique_complex(nx.complete_graph(4))
    >>> cc.dimension()
    3
    """
    facets = nx.find_cliques(graph) if graph else iter([[]])
    return SimplicialComplex._from_maximal_faces(graph, facets)


def nerve_of_cliques(graph: nx.Graph) -> SimplicialComplex:
    """Return the nerve of the maximal cliques of *graph*.

    Pairwise intersection determines the 1-skeleton, but a simplex in the
    nerve requires a common vertex shared by every participating maximal
    clique.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import nerve_of_cliques
    >>> n = nerve_of_cliques(nx.cycle_graph(4))
    >>> n.dimension()
    1
    """
    the_cliques = {frozenset(q) for q in nx.find_cliques(graph)}
    return nerve_of_sets(the_cliques)


def bounded_degree(
    graph: nx.Graph, lambda_vector: dict, list_of_edges: Iterable
) -> bool:
    """Return whether the edge-induced subgraph respects degree bounds.

    For every vertex *v* in the subgraph induced by *list_of_edges*, check
    that its degree does not exceed ``lambda_vector[v]``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import bounded_degree
    >>> g = nx.path_graph(3)
    >>> bounded_degree(g, {0: 1, 1: 1, 2: 1}, [(0, 1)])
    True
    >>> bounded_degree(g, {0: 1, 1: 1, 2: 1}, [(0, 1), (1, 2)])
    False
    """
    subgraph = graph.edge_subgraph(list_of_edges)
    for v in graph:
        if v in subgraph.nodes() and subgraph.degree(v) > lambda_vector[v]:
            return False
    return True


def bounded_degree_complex(graph: nx.Graph, lambda_vector: dict) -> SimplicialComplex:
    """Return the bounded-degree complex of *graph*.

    The simplices are subsets of edges whose induced subgraph satisfies the
    degree bounds given by *lambda_vector*.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import bounded_degree_complex
    >>> g = nx.cycle_graph(3)
    >>> lv = {0: 1, 1: 1, 2: 1}
    >>> bdc = bounded_degree_complex(g, lv)
    >>> bdc.dimension()
    0
    """

    def _bounded(s: set) -> bool:
        return bounded_degree(graph, lambda_vector, s)

    return SimplicialComplex(graph.edges(), function=_bounded)


def is_oriented_simplex(digraph: nx.DiGraph) -> bool:
    """Return whether *digraph* is an oriented simplex.

    A directed graph is an oriented simplex when it is both a DAG and a
    tournament (a complete directed graph with no symmetric edges).

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import is_oriented_simplex
    >>> d = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    >>> is_oriented_simplex(d)
    True
    >>> d2 = nx.DiGraph([(0, 1), (1, 0)])
    >>> is_oriented_simplex(d2)
    False
    """
    return nx.is_directed_acyclic_graph(digraph) and bool(
        tournament.is_tournament(digraph)
    )


def oriented_complex(digraph: nx.DiGraph) -> SimplicialComplex:
    """Return the oriented complex of a directed graph.

    The oriented complex has the nodes of *digraph* as vertices and its
    simplices are the subsets that induce an oriented simplex (a tournament
    that is also a DAG).

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import oriented_complex
    >>> d = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    >>> oc = oriented_complex(d)
    >>> oc.dimension()
    2
    """

    def _oriented_simplex(s: set) -> bool:
        return is_oriented_simplex(digraph.subgraph(s))

    return SimplicialComplex(digraph.nodes(), function=_oriented_simplex)


def complex_of_forests(
    graph: nx.Graph, max_deg: int | float = math.inf
) -> SimplicialComplex:
    """Return the forest complex of *graph*.

    The simplices are subsets of vertices that induce a forest whose maximum
    degree does not exceed *max_deg*.

    .. rubric:: Parameters

    graph : networkx.Graph
        The input graph.
    max_deg : int or float, optional
        Maximum allowed degree in the induced forest (default: ``math.inf``).

    .. rubric:: Returns

    SimplicialComplex
        The complex of forests.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import complex_of_forests
    >>> g = nx.cycle_graph(4)
    >>> cf = complex_of_forests(g)
    >>> cf.dimension()
    2
    >>> cf_deg1 = complex_of_forests(g, max_deg=1)
    >>> cf_deg1.dimension()
    1
    """

    def _is_forest(s: set) -> bool:
        if not s:  # pragma: no cover
            return True
        subgraph = graph.subgraph(s)
        # Prevent max() empty sequence error by setting default=0
        maxd = max([subgraph.degree(node) for node in subgraph.nodes], default=0)
        return nx.is_forest(subgraph) and maxd <= max_deg

    return SimplicialComplex(graph.nodes(), function=_is_forest)


def neighborhood_complex(graph: nx.Graph) -> SimplicialComplex:
    """Return the neighborhood complex of *graph* as defined by Lovász.

    The neighborhood complex has the vertices of *graph* as its vertex set.
    The simplices are the subsets of vertices that share a common neighbor,
    i.e., subsets ``s`` such that there exists a vertex ``v`` adjacent to
    every vertex in ``s``.  The facets are exactly the open neighborhoods
    ``N(v)`` for each vertex ``v``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import neighborhood_complex
    >>> nc = neighborhood_complex(nx.cycle_graph(5))
    >>> nc.dimension()
    1
    >>> nc = neighborhood_complex(nx.complete_graph(4))
    >>> nc.dimension()
    2
    """
    facets = [Simplex(graph.neighbors(v)) for v in graph.nodes()]
    facets = [f for f in facets if f]  # drop empty neighborhoods
    return SimplicialComplex(graph.nodes(), facet_set=facets)


def directed_neighborhood_complex(
    digraph: nx.DiGraph, use_out_neighborhood: bool = False
) -> SimplicialComplex:
    """Return the directed neighborhood complex of *digraph*.

    The vertex set is the node set of *digraph*.  By default the facets are
    the open **in**-neighborhoods ``N^-(v) = {u : (u, v) in E}`` for each
    vertex *v*.  When *use_out_neighborhood* is ``True`` the facets are the
    open **out**-neighborhoods ``N^+(v) = {u : (v, u) in E}`` instead.

    .. rubric:: Parameters

    digraph : networkx.DiGraph
        The input directed graph.
    use_out_neighborhood : bool, optional
        When ``True``, use out-neighborhoods as facets (default: ``False``).

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import directed_neighborhood_complex
    >>> d = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    >>> nc = directed_neighborhood_complex(d)
    >>> nc.dimension()
    1
    >>> nc_out = directed_neighborhood_complex(d, use_out_neighborhood=True)
    >>> nc_out.dimension()
    1
    """
    if use_out_neighborhood:
        facets = [Simplex(digraph.successors(v)) for v in digraph.nodes()]
    else:
        facets = [Simplex(digraph.predecessors(v)) for v in digraph.nodes()]
    facets = [f for f in facets if f]  # drop empty neighborhoods
    return SimplicialComplex(digraph.nodes(), facet_set=facets)


# ---------------------------------------------------------------------------
# Star, link and deletion of a general simplex
# ---------------------------------------------------------------------------


def _as_simplicial_complex(
    complex_or_graph: SimplicialComplex | nx.Graph,
) -> SimplicialComplex:
    """Coerce *complex_or_graph* into a :class:`SimplicialComplex`.

    A :class:`networkx.Graph` is interpreted as its clique complex.
    """
    if isinstance(complex_or_graph, SimplicialComplex):
        return complex_or_graph
    return clique_complex(complex_or_graph)


def _sigma_to_simplex(sigma) -> Simplex:
    """Normalize *sigma* (a single vertex or a set of vertices) to a Simplex."""
    if isinstance(sigma, Simplex):
        return sigma
    if isinstance(sigma, (frozenset, set, tuple, list)):
        return Simplex(sigma)
    return Simplex({sigma})


def star(complex_or_graph: SimplicialComplex | nx.Graph, sigma) -> SimplicialComplex:
    """Return the closed star of the simplex *sigma*.

    ``st(sigma) = {tau in complex : sigma union tau is a simplex}``, realized
    here as the union of every facet that contains *sigma*.  *sigma* may be a
    single vertex or a set of vertices; *complex_or_graph* may be a
    :class:`SimplicialComplex` or a :class:`networkx.Graph` (interpreted as
    its clique complex).

    .. rubric:: Examples

    >>> from pycombtop import SimplicialComplex, star
    >>> sc = SimplicialComplex({0, 1, 2, 3}, facet_set=[{0, 1, 2}, {0, 3}])
    >>> sorted(star(sc, 0).vertex_set)
    [0, 1, 2, 3]
    >>> sorted(star(sc, {1, 2}).vertex_set)
    [0, 1, 2]
    """
    sc = _as_simplicial_complex(complex_or_graph)
    sigma = _sigma_to_simplex(sigma)
    facets = {f for f in sc.facet_set if sigma <= f}
    if not facets:
        return SimplicialComplex(set(), facet_set=set())
    vertices = set.union(*(set(s) for s in facets))
    return SimplicialComplex(vertices, facet_set=facets)


def link(complex_or_graph: SimplicialComplex | nx.Graph, sigma) -> SimplicialComplex:
    """Return the link of the simplex *sigma*.

    ``lk(sigma) = {tau in complex : tau disjoint from sigma, tau union sigma
    in complex}``.  *sigma* may be a single vertex or a set of vertices, and
    is computed by iterating :meth:`SimplicialComplex.link` over the
    vertices of *sigma* (the result does not depend on the order).
    *complex_or_graph* may be a :class:`SimplicialComplex` or a
    :class:`networkx.Graph` (interpreted as its clique complex).

    .. rubric:: Examples

    >>> from pycombtop import SimplicialComplex, link
    >>> sc = SimplicialComplex({0, 1, 2, 3}, facet_set=[{0, 1, 2}, {0, 1, 3}])
    >>> sorted(link(sc, {0, 1}).vertex_set)
    [2, 3]
    """
    sc = _as_simplicial_complex(complex_or_graph)
    sigma = _sigma_to_simplex(sigma)
    result = sc
    for v in sigma:
        result = result.link(v)
    return result


def deletion(
    complex_or_graph: SimplicialComplex | nx.Graph, sigma
) -> SimplicialComplex:
    """Return the deletion of the simplex *sigma* from the ambient complex.

    ``deletion(sigma) = {tau in complex : sigma not subset of tau}``.  This
    keeps every vertex of the ambient complex and removes only the top
    simplices that pass through *sigma* -- it is **not** the same as
    deleting *sigma*'s vertices.

    This operation is the simplicial-complex analogue of removing the closed
    star of a simplex while keeping the ambient vertex set unchanged.  It is a
    general-purpose construction used when one wants to remove all faces that
    contain *sigma* without changing the underlying vertex set.

    .. rubric:: Care point

    The deletion is generally **not flag** (not the clique complex of any
    graph), even when the ambient complex is.  It is therefore always built as
    a genuine :class:`SimplicialComplex` via the membership-function
    constructor rather than via any graph-based (flag) shortcut; building it
    from graph tools can silently produce a wrong answer.  Use
    :func:`~pycombtop.homotopy_type.homotopy_type_sc_with_verdict` (not
    :func:`~pycombtop.homotopy_type.homotopy_type_with_verdict`) to compute
    its homotopy type.

    .. rubric:: Examples

    A hollow triangle glued to a filled one at a single vertex: the deletion
    of the filled triangle's apex is the hollow triangle, which is not flag
    (its 1-skeleton has a triangle that isn't a 2-simplex of the complex).

    >>> from pycombtop import SimplicialComplex, deletion
    >>> sc = SimplicialComplex(
    ...     {0, 1, 2, 3, 4},
    ...     facet_set=[{0, 1, 2}, {2, 3}, {3, 4}, {2, 4}],
    ... )
    >>> d = deletion(sc, {0})
    >>> d.is_clique_complex()
    False
    >>> sorted(d.vertex_set)
    [1, 2, 3, 4]
    """
    sc = _as_simplicial_complex(complex_or_graph)
    sigma = _sigma_to_simplex(sigma)

    def _member(s) -> bool:
        s = Simplex(s)
        return sc.function(s) and not sigma <= s

    return SimplicialComplex(sc.base_set, function=_member)
