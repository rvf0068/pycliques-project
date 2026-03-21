from __future__ import annotations

import math
from collections.abc import Callable, Generator, Iterable
from functools import reduce
from itertools import chain, combinations

import networkx as nx
from networkx.algorithms import tournament


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
    """A simplicial complex over a finite vertex set.

    A :class:`SimplicialComplex` is composed of a set of vertices and a set of
    simplices (of type :class:`Simplex`), which correspond to subsets of the
    vertex set.  It can be constructed either from an explicit set of facets or
    from a membership function that decides which subsets are simplices.

    .. rubric:: Parameters

    vertex_set : set
        The ground set of vertices.
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
        vertex_set: Iterable,
        facet_set: Iterable | None = None,
        function: Callable[[set], bool] | None = None,
    ) -> None:
        self.vertex_set = set(vertex_set)

        # Determine the membership function
        if function is not None:
            self.function = function
        else:
            self.function = self._default_is_simplex

        # Determine the facets
        if facet_set is not None:
            self.facet_set = {Simplex(s) for s in facet_set}
        else:
            self.facet_set = self._facet_set_from_function()

    def _default_is_simplex(self, s: set) -> bool:
        """Fallback membership function if none is provided."""
        if not hasattr(self, "facet_set") or self.facet_set is None:
            return False
        return any(s <= facet for facet in self.facet_set)

    def _facet_set_from_function(self) -> set[Simplex]:
        """Derive facets using the membership function.

        This calculates the facet set by finding the maximal subsets
        of vertices that satisfy *self.function*.
        """
        if self.function(self.vertex_set):
            return {Simplex(self.vertex_set)}

        facets = set()
        # Evaluate subsets lazily to save memory
        for r in reversed(range(1, len(self.vertex_set) + 1)):
            for subset in combinations(self.vertex_set, r):
                s = Simplex(subset)
                if self.function(s):
                    # Check if s is already subsumed by a known facet
                    if not any(s.issubset(f) for f in facets):
                        facets.add(s)
        return facets

    def __repr__(self) -> str:
        base = f"Simplicial complex with vertex_set {self.vertex_set} "
        return base + f"and facets {self.facet_set}."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimplicialComplex):
            return False

        return self.vertex_set == other.vertex_set and self.facet_set == other.facet_set

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
        >>> 2 in d.vertex_set
        False
        """
        vertices = self.vertex_set - {x}
        containing = {f for f in self.facet_set if x in f}
        not_containing = self.facet_set - containing

        good_facets = set(not_containing)
        for s in containing:
            candidate = s - {x}
            # Only add if it's not strictly contained in another existing facet
            if not any(candidate.issubset(f) for f in not_containing):
                good_facets.add(candidate)

        return SimplicialComplex(vertices, facet_set=good_facets)

    def link(self, x) -> SimplicialComplex:
        """Return the link of vertex *x* in the complex.

        The link consists of all simplices *s* such that *s* does not contain
        *x* but *s* ∪ {*x*} is a simplex of the complex.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
        >>> lk = sc.link(0)
        >>> lk.dimension()
        1
        >>> 0 in lk.vertex_set
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

        def _new_function(s: set) -> bool:
            return self.function(s) and len(s) <= n + 1

        return SimplicialComplex(self.vertex_set, function=_new_function)

    def one_skeleton_graph(self) -> nx.Graph:
        """Return the 1-skeleton as a :class:`networkx.Graph`.

        Vertices become graph nodes and 1-simplices become edges.

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
        edges = [
            (i, j)
            for (i, j) in combinations(self.vertex_set, 2)
            if self.function({i, j})
        ]
        the_graph.add_edges_from(edges)
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
        return self == clique_complex(self.one_skeleton_graph())

    def all_simplices(self) -> set[Simplex]:
        """Return the set of all simplices in the complex.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
        >>> sorted(len(s) for s in sc.all_simplices())
        [0, 1, 1, 2]
        """
        all_simplices_set = set()
        for facet in self.facet_set:
            s = list(facet)
            # chain.from_iterable creates a generator, set() consumes it.
            subset_generator = chain.from_iterable(
                combinations(s, r) for r in range(len(s) + 1)
            )
            all_simplices_set.update(subset_generator)
        return {Simplex(s) for s in all_simplices_set}

    def dong_matching(
        self, order_function: Callable[[set], list] = list
    ) -> set[Simplex]:
        """Return the critical simplices under Dong's matching.

        The matching pairs simplices with their unions with a vertex according
        to the vertex ordering given by *order_function*.  The returned set
        contains exactly the unmatched (critical) simplices.

        .. rubric:: Parameters

        order_function : callable, optional
            A function that converts the vertex set into an ordered list
            (default: ``list``).

        .. rubric:: Returns

        set of :class:`Simplex`
            The unmatched (critical) simplices.

        .. rubric:: Examples

        >>> from pycombtop import SimplicialComplex
        >>> sc = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
        >>> critical = sc.dong_matching(order_function=sorted)
        >>> len(critical) % 2  # critical simplices are not paired
        0
        """
        matched = set()
        vertices = order_function(self.vertex_set)

        for vertex in vertices:
            the_link = self.link(vertex)
            link_simplices = the_link.all_simplices()

            for s in link_simplices:
                s_plus_v = s | {vertex}
                if (s not in matched) and (s_plus_v not in matched):
                    matched.add(s)
                    matched.add(s_plus_v)

        return self.all_simplices() - matched


# --- Module Functions ---


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


def nerve_of_sets(sets: Iterable[set]) -> SimplicialComplex:
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

    def _non_empty_intersection(s: set) -> bool:
        if not s:
            return False
        intersect = reduce(lambda x, y: x.intersection(y), list(s))
        return len(intersect) != 0

    vertices = [Simplex(s) for s in sets]
    return SimplicialComplex(vertices, function=_non_empty_intersection)


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
    the_cliques = {Simplex(q) for q in nx.find_cliques(graph)}
    return SimplicialComplex(graph.nodes(), facet_set=the_cliques)


def nerve_of_cliques(graph: nx.Graph) -> SimplicialComplex:
    """Return the nerve of the maximal cliques of *graph*.

    Two cliques are connected in the nerve when they share at least one vertex.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop import nerve_of_cliques
    >>> n = nerve_of_cliques(nx.cycle_graph(4))
    >>> n.dimension()
    1
    """
    the_cliques = {frozenset(q) for q in nx.find_cliques(graph)}
    return nerve_of_sets(the_cliques)


def bounded_degree(graph: nx.Graph, lambda_vector: dict, list_of_edges: list) -> bool:
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
    return nx.is_directed_acyclic_graph(digraph) and tournament.is_tournament(digraph)


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
        if not s:  # Handle the empty face safely
            return True
        subgraph = graph.subgraph(s)
        # Prevent max() empty sequence error by setting default=0
        maxd = max([subgraph.degree(node) for node in subgraph.nodes], default=0)
        return nx.is_forest(subgraph) and maxd <= max_deg

    return SimplicialComplex(graph.nodes(), function=_is_forest)
