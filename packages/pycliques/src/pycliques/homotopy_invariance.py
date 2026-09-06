r"""Machinery for attacking :math:`G \simeq K(G)` ("G is good") conjectures.

This module collects four related pieces of machinery that recurred, ad hoc,
across a research session working on open conjectures about the clique graph
operator :math:`K`. They all share the same underlying objects: completes of
:math:`K(G)` (pairwise-intersecting sets of cliques of :math:`G`), and their
intersections/centers/closures. See ``research/homotopy-invariance-toolkit.md``
for the full narrative and ``research/references/`` for the cited papers,
in particular the 2008 poset paper (Larrion, Pizana, Villarroel-Flores,
"Posets, Clique Graphs and their Homotopy Type," *European J. Combin.* 29)
for Theorems 11/13/15, and Islas-Gomez's 2021 thesis for the elementary
collapse dichotomy referenced in Priority 4 of that document.

.. rubric:: Note on package placement

:func:`theorem15_hypothesis_holds` needs a homotopy-type computation from
:mod:`pycombtop`, which itself depends on :mod:`pycliques`. To avoid a
circular *package* dependency, that one function imports
:mod:`pycombtop.homotopy_type` lazily (inside the function body) rather than
at module load time; every other function here only needs :mod:`networkx`
and :mod:`pycliques.cliques`.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterable, Iterator

import networkx as nx

from .cliques import clique_graph

# ---------------------------------------------------------------------------
# 1a. Completes of K(G), centers, and neckties (Theorem 11 machinery)
# ---------------------------------------------------------------------------


def completes_with_empty_intersection(
    kg: nx.Graph, max_size: int | None = None
) -> Iterator[frozenset]:
    """Yield every complete of *kg* whose total intersection is empty.

    A "complete" of :math:`K(G)` is a set of pairwise-intersecting cliques of
    :math:`G`, i.e. a clique of the clique graph *kg*. A complete is *bad*
    when the cliques' total intersection is empty.

    .. rubric:: Parameters

    kg : networkx.Graph
        A clique graph, e.g. the output of :func:`pycliques.cliques.clique_graph`.
    max_size : int, optional
        When given, stop once completes exceed this size (relies on
        :func:`networkx.enumerate_all_cliques` yielding cliques in
        non-decreasing size order).

    .. rubric:: Returns

    Iterator[frozenset]
        Each bad complete, as a ``frozenset`` of cliques (nodes of *kg*).

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cliques import clique_graph
    >>> from pycliques.homotopy_invariance import completes_with_empty_intersection
    >>> kg = clique_graph(nx.octahedral_graph())
    >>> bad = list(completes_with_empty_intersection(kg))
    >>> len(bad) > 0
    True
    """
    for c in nx.enumerate_all_cliques(kg):
        if len(c) < 2:
            continue
        if max_size is not None and len(c) > max_size:
            break
        inter = set.intersection(*(set(q) for q in c))
        if not inter:
            yield frozenset(c)


def completes_of_size(kg: nx.Graph, k: int) -> Iterator[frozenset]:
    """Yield every complete of *kg* (good or bad) with exactly *k* elements.

    .. rubric:: Parameters

    kg : networkx.Graph
        A clique graph.
    k : int
        The desired complete size.

    .. rubric:: Returns

    Iterator[frozenset]
        Each size-*k* complete (clique of *kg*).

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cliques import clique_graph
    >>> from pycliques.homotopy_invariance import completes_of_size
    >>> kg = clique_graph(nx.complete_graph(4))
    >>> list(completes_of_size(kg, 1))
    [frozenset({{0, 1, 2, 3}})]
    """
    if k < 1:
        return
    for c in nx.enumerate_all_cliques(kg):
        if len(c) < k:
            continue
        if len(c) > k:
            break
        yield frozenset(c)


def neckties(kg: nx.Graph) -> list[frozenset]:
    """Return the neckties of *kg*: maximal bad completes.

    .. rubric:: Parameters

    kg : networkx.Graph
        A clique graph.

    .. rubric:: Returns

    list[frozenset]
        Each maximal (by inclusion) bad complete of *kg*.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cliques import clique_graph
    >>> from pycliques.homotopy_invariance import neckties
    >>> kg = clique_graph(nx.octahedral_graph())
    >>> len(neckties(kg)) > 0
    True
    """
    result = []
    for c in nx.find_cliques(kg):
        inter = set.intersection(*(set(q) for q in c)) if c else set()
        if not inter:
            result.append(frozenset(c))
    return result


def is_center(q0: Iterable, complete: Iterable) -> bool:
    """Return whether *q0* is a center of the bad complete *complete*.

    For a bad complete :math:`X`, :math:`q_0` is a *center* if for every
    :math:`Y \\subseteq X` with :math:`\\bigcap Y \\neq \\emptyset`,
    :math:`q_0 \\cap \\bigcap Y \\neq \\emptyset` too.

    .. rubric:: Parameters

    q0 : Hashable
        A clique of :math:`G` (a node of :math:`K(G)`), candidate center.
    complete : Iterable
        The bad complete :math:`X` (an iterable of cliques of :math:`G`).

    .. rubric:: Returns

    bool
        ``True`` if *q0* is a center of *complete*.

    .. rubric:: Examples

    >>> from pycliques.cliques import clique_graph
    >>> from pycliques.named import octahedron
    >>> from pycliques.homotopy_invariance import (
    ...     completes_with_empty_intersection, is_center)
    >>> kg = clique_graph(octahedron(3))
    >>> X = next(completes_with_empty_intersection(kg))
    >>> sorted(X, key=sorted)
    [{0, 2, 4}, {0, 3, 5}, {1, 2, 5}]
    >>> is_center(frozenset({0, 2, 5}), X)
    True
    >>> is_center(frozenset({0, 3, 4}), X)
    False
    """
    members = list(complete)
    q0_set = set(q0)
    for r in range(1, len(members) + 1):
        for sub in itertools.combinations(members, r):
            inter_sub = set.intersection(*(set(q) for q in sub))
            if inter_sub and not (inter_sub & q0_set):
                return False
    return True


def has_center_at_all(complete: Iterable, kg: nx.Graph) -> bool:
    """Return whether *complete* has a center among the nodes of *kg*.

    Unlike :func:`is_center`, which checks a single candidate, this checks
    every node of *kg* and drops any necktie-membership requirement.

    .. rubric:: Parameters

    complete : Iterable
        The bad complete to test.
    kg : networkx.Graph
        The clique graph whose nodes are candidate centers.

    .. rubric:: Returns

    bool
        ``True`` if some node of *kg* is a center of *complete*.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cliques import clique_graph
    >>> from pycliques.named import octahedron
    >>> from pycliques.homotopy_invariance import (
    ...     completes_with_empty_intersection, has_center_at_all)
    >>> kg = clique_graph(octahedron(3))
    >>> bad = next(completes_with_empty_intersection(kg))
    >>> isinstance(has_center_at_all(bad, kg), bool)
    True
    """
    return any(is_center(q0, complete) for q0 in kg.nodes())


def theorem11_hypothesis_holds(
    graph: nx.Graph,
) -> tuple[bool, frozenset | None]:
    """Check Theorem 11's hypothesis: every bad complete has a necktie-wide center.

    Every bad complete :math:`X` of :math:`K(G)` must have a center lying in
    every necktie containing :math:`X`.

    .. rubric:: Parameters

    graph : networkx.Graph
        The graph :math:`G` to test.

    .. rubric:: Returns

    tuple[bool, frozenset | None]
        ``(True, None)`` if the hypothesis holds for every bad complete;
        otherwise ``(False, X)`` where *X* is a bad complete without a
        qualifying center.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.homotopy_invariance import theorem11_hypothesis_holds
    >>> holds, _ = theorem11_hypothesis_holds(nx.complete_graph(4))
    >>> holds
    True
    """
    kg = clique_graph(graph)
    assert kg is not None
    nt = neckties(kg)
    for complete in completes_with_empty_intersection(kg):
        containing = [necktie for necktie in nt if complete <= necktie]
        if not any(
            is_center(q0, complete) and all(q0 in necktie for necktie in containing)
            for q0 in kg.nodes()
        ):
            return False, complete
    return True, None


# ---------------------------------------------------------------------------
# 1b. Minimal violation and witness construction
# ---------------------------------------------------------------------------


def _total_intersection(complete: Iterable) -> frozenset:
    """Return the total intersection of a complete, as a frozenset."""
    sets = [set(q) for q in complete]
    if not sets:
        return frozenset()
    return frozenset(set.intersection(*sets))


def minimal_violation(kg: nx.Graph) -> tuple | None:
    """Return a smallest bad complete of *kg* (a clique graph), or ``None``.

    A smallest bad complete forces every proper subset of it to be good
    (nonempty total intersection): if a proper subset were bad too, it would
    be a smaller bad complete, contradicting minimality.

    Builds up completes level by level starting from pairs (edges of *kg*,
    always good, since two intersecting cliques already share a vertex),
    only ever extending complete-so-far candidates that are themselves good
    -- rather than testing every size cold. Goodness (nonempty intersection)
    is *downward closed*: any subset of a good complete is good too, because
    intersecting fewer sets can only enlarge (or preserve) the intersection.
    Equivalently, badness is *upward closed*. This means that to confirm a
    candidate is a minimal violation, it is enough to check that every
    *omit-one* subset is good -- every smaller subset is then automatically
    good too, since it is a subset of some omit-one subset.

    .. rubric:: Parameters

    kg : networkx.Graph
        A clique graph.

    .. rubric:: Returns

    tuple | None
        A smallest bad complete (as a tuple of cliques), or ``None`` if *kg*
        is clique-Helly (has no bad completes at all).

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cliques import clique_graph
    >>> from pycliques.homotopy_invariance import minimal_violation
    >>> kg = clique_graph(nx.complete_graph(4))
    >>> minimal_violation(kg) is None
    True
    >>> kg = clique_graph(nx.octahedral_graph())
    >>> len(minimal_violation(kg))
    3
    """
    node_sets = {q: set(q) for q in kg.nodes()}
    frontier: list[tuple[frozenset, frozenset]] = [
        (frozenset((u, v)), frozenset(node_sets[u] & node_sets[v]))
        for u, v in kg.edges()
    ]
    while frontier:
        next_frontier: list[tuple[frozenset, frozenset]] = []
        next_seen: set[frozenset] = set()
        for complete, inter in frontier:
            common_neighbors: set | None = None
            for q in complete:
                nbrs = set(kg.neighbors(q))
                common_neighbors = (
                    nbrs if common_neighbors is None else common_neighbors & nbrs
                )
            if not common_neighbors:
                continue
            for q in common_neighbors - complete:
                candidate = complete | {q}
                if candidate in next_seen:
                    continue
                new_inter = inter & node_sets[q]
                if not new_inter:
                    if all(_total_intersection(candidate - {y}) for y in candidate):
                        return tuple(candidate)
                    continue
                next_seen.add(candidate)
                next_frontier.append((candidate, frozenset(new_inter)))
        frontier = next_frontier
    return None


def witnesses_and_extension(
    graph: nx.Graph, kg: nx.Graph, violation: Iterable
) -> tuple[list, list]:
    """Return witnesses for a minimal violation, and their clique extension.

    Given a minimal bad complete :math:`Z = (q_1, \\dots, q_m)`, for each
    :math:`i` the *witness* :math:`x_i \\in \\bigcap_{k \\neq i} q_k` always
    exists (by minimality, the omit-one sub-complete is good), is pairwise
    distinct from the other witnesses, and the witnesses are pairwise
    adjacent in :math:`G` -- so they extend to a clique :math:`q^*`, which is
    provably distinct from every element of :math:`Z`.

    .. rubric:: Parameters

    graph : networkx.Graph
        The graph :math:`G`.
    kg : networkx.Graph
        The clique graph :math:`K(G)`; used only to validate that *violation*
        consists of nodes of *kg*.
    violation : Iterable
        A minimal bad complete of *kg* (e.g. from :func:`minimal_violation`).

    .. rubric:: Returns

    tuple[list, list]
        ``(witnesses, q_star)``: the list of witnesses :math:`x_i`, and a
        clique :math:`q^*` of *graph* extending them.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cliques import clique_graph
    >>> from pycliques.homotopy_invariance import (
    ...     minimal_violation, witnesses_and_extension)
    >>> g = nx.octahedral_graph()
    >>> kg = clique_graph(g)
    >>> z = minimal_violation(kg)
    >>> witnesses, q_star = witnesses_and_extension(g, kg, z)
    >>> len(witnesses) == len(z)
    True
    >>> set(witnesses) <= set(q_star)
    True
    """
    members = list(violation)
    if any(q not in kg for q in members):
        raise ValueError("violation must consist of nodes (cliques) of kg")
    m = len(members)
    witnesses = [
        next(iter(set.intersection(*(set(members[j]) for j in range(m) if j != i))))
        for i in range(m)
    ]
    q_star = next(q for q in nx.find_cliques(graph) if set(witnesses) <= set(q))
    return witnesses, q_star


# ---------------------------------------------------------------------------
# 1d. The h-closure operator and Theorem 15's direct hypothesis test
# ---------------------------------------------------------------------------


def h_closure(
    complete: Iterable,
    kg: nx.Graph,
    all_maximal_cliques_of_kg: Iterable[frozenset],
) -> frozenset:
    """Return :math:`h(X)`: the intersection of every maximal clique of
    :math:`K(K(G))` containing X.

    Defined for a complete :math:`X` of :math:`K(G)` as the intersection of
    all cliques of :math:`K(K(G))` (i.e. maximal completes of :math:`K(G)`)
    containing :math:`X`. Idempotent: ``h_closure(h_closure(X, ...), ...)
    == h_closure(X, ...)``.

    .. rubric:: Parameters

    complete : Iterable
        A complete of *kg* (an iterable of cliques of :math:`G`).
    kg : networkx.Graph
        The clique graph :math:`K(G)`. Present for API symmetry with
        :func:`theorem15_hypothesis_holds`; not otherwise used since
        *all_maximal_cliques_of_kg* already carries the needed information.
    all_maximal_cliques_of_kg : Iterable[frozenset]
        The maximal cliques of *kg* (e.g. from ``nx.find_cliques(kg)``).

    .. rubric:: Returns

    frozenset
        :math:`h(X)`, a subset of ``kg``'s nodes.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cliques import clique_graph
    >>> from pycliques.homotopy_invariance import h_closure
    >>> g = nx.complete_graph(4)
    >>> kg = clique_graph(g)
    >>> all_maximal = [frozenset(c) for c in nx.find_cliques(kg)]
    >>> h_closure(frozenset(kg.nodes()), kg, all_maximal) == frozenset(kg.nodes())
    True
    """
    del kg  # unused: kept for API symmetry, see docstring
    complete = frozenset(complete)
    containing = [q for q in all_maximal_cliques_of_kg if complete <= q]
    if not containing:
        return complete
    return frozenset.intersection(*containing)


def delta_of_clique_set(clique_set: Iterable[frozenset]) -> nx.Graph:
    """Return :math:`\\Delta(C)`: one full simplex per clique in *clique_set*.

    The result lives on :math:`G`'s vertices (not on cliques): it is the
    union, over every clique :math:`q` in *clique_set*, of the complete
    graph on :math:`q`'s vertices.

    .. rubric:: Parameters

    clique_set : Iterable[frozenset]
        A set of cliques of :math:`G` (e.g. ``h(X)`` from :func:`h_closure`).

    .. rubric:: Returns

    networkx.Graph
        A graph whose clique complex is :math:`\\Delta(C)`.

    .. rubric:: Examples

    >>> from pycliques.homotopy_invariance import delta_of_clique_set
    >>> d = delta_of_clique_set([frozenset({0, 1, 2})])
    >>> sorted(d.nodes())
    [0, 1, 2]
    >>> d.number_of_edges()
    3
    """
    clique_set = list(clique_set)
    verts: set = set()
    for q in clique_set:
        verts |= set(q)
    delta = nx.Graph()
    delta.add_nodes_from(verts)
    for q in clique_set:
        for u, v in itertools.combinations(q, 2):
            delta.add_edge(u, v)
    return delta


def _maximal_clique_index(
    all_maximal: Iterable[frozenset],
) -> dict[Hashable, set[frozenset]]:
    """Index maximal cliques of kg by the (clique-of-G) nodes they contain."""
    index: dict[Hashable, set[frozenset]] = {}
    for q in all_maximal:
        for node in q:
            index.setdefault(node, set()).add(q)
    return index


def _h_closure_indexed(
    complete: Iterable,
    index: dict[Hashable, set[frozenset]],
    cache: dict[frozenset, frozenset],
) -> frozenset:
    """Optimized h(X): use a per-node index instead of scanning every maximal clique."""
    complete = frozenset(complete)
    cached = cache.get(complete)
    if cached is not None:
        return cached
    candidate_sets = [index.get(node, set()) for node in complete]
    if not candidate_sets:
        result = complete
    else:
        containing = set.intersection(*candidate_sets)
        result = frozenset.intersection(*containing) if containing else complete
    cache[complete] = result
    return result


def theorem15_hypothesis_holds(
    graph: nx.Graph, size_cap: int | None = None
) -> tuple[bool | None, frozenset | None]:
    """Check Theorem 15's hypothesis: :math:`\\Delta(h(X))` contractible for every X.

    If :math:`\\Delta(h(X))` is contractible for every complete :math:`X` of
    :math:`K(G)`, then :math:`G \\simeq K(G)`. This is an application of
    Quillen's fiber lemma (survey Theorem 13/15, from the 2008 poset paper)
    and, importantly, does not require exhibiting an explicit combinatorial
    collapse -- so it is not vulnerable to a Whitehead-torsion-style
    obstruction the way elementary-collapse-based approaches are.

    By Proposition 14 of the survey, it is provably enough to check
    completes with ``h(X) == X`` (:math:`h`'s fixed points); since :math:`h`
    is idempotent, the *distinct* values of ``h_closure(X, ...)`` over all
    completes :math:`X` are exactly this set of fixed points, so no separate
    filtering step is needed. To keep the per-complete cost of computing
    :math:`h` down (a linear scan over every maximal clique, for every one
    of potentially thousands of completes, was the confirmed bottleneck),
    this indexes maximal cliques by the nodes they contain and memoizes
    results, rather than scanning ``all_maximal_cliques_of_kg`` from
    scratch for each complete.

    .. rubric:: Parameters

    graph : networkx.Graph
        The graph :math:`G` to test.
    size_cap : int, optional
        Skip (and report as inconclusive) any :math:`\\Delta(h(X))` with more
        than this many vertices, rather than risking a very slow
        homotopy-type computation.

    .. rubric:: Returns

    tuple[bool | None, frozenset | None]
        ``(True, None)`` if the hypothesis holds for every complete;
        ``(False, hX)`` if some :math:`\\Delta(h(X))` is provably not
        contractible; ``(None, hX)`` if *size_cap* caused an early,
        inconclusive stop.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.homotopy_invariance import theorem15_hypothesis_holds
    >>> theorem15_hypothesis_holds(nx.complete_graph(4))
    (True, None)
    """
    from pycombtop.homotopy_type import homotopy_type_with_verdict

    kg = clique_graph(graph)
    assert kg is not None
    all_maximal = [frozenset(c) for c in nx.find_cliques(kg)]
    index = _maximal_clique_index(all_maximal)
    cache: dict[frozenset, frozenset] = {}

    distinct_h: set[frozenset] = set()
    for complete in nx.enumerate_all_cliques(kg):
        distinct_h.add(_h_closure_indexed(complete, index, cache))

    for h_image in distinct_h:
        delta = delta_of_clique_set(h_image)
        if size_cap is not None and delta.order() > size_cap:
            return None, h_image
        verdict = homotopy_type_with_verdict(delta)
        if not (verdict.is_exact and verdict.wedge.is_contractible()):
            return False, h_image
    return True, None
