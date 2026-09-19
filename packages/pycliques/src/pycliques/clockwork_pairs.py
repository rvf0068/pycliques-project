r"""Maps of coaffine pairs between clockwork graphs and target graphs.

This module implements the classifier criterion of Theorem 2.6 of

    F. Larrion, V. Neumann-Lara, M. A. Pizana,
    *Graph relations, clique divergence and surface triangulations* (2006).

Theorem 2.6 states that an admissible graph relation between :math:`r`-coaffine
graphs transfers rank divergence from the domain to the codomain: if
:math:`f\colon (A,\alpha)\to(B,\beta)` satisfies :math:`f\circ\alpha=\beta\circ f`
and both pairs are :math:`r`-coaffine, then rank divergence of :math:`A` implies
rank divergence of :math:`B`.

Theorem 3.1 of the same paper states that the clockwork graph
:math:`R_{2m}^n` (see :func:`r_clock`), together with its canonical
automorphism :math:`\sigma` (see :func:`canonical_clockwork_coaffination`), is
:math:`(m+1)`-coaffine and rank divergent.  The paper defines :math:`R_{2m}^n`
-- and therefore asserts Theorem 3.1 -- only for :math:`m\geq 2`; a classifier
search based on this theorem must never use ``m == 1`` (radius 2), since
Theorem 3.1 gives no rank-divergence guarantee in that case.  (This module's
low-level constructors accept ``m == 1`` anyway, since the construction
itself and its ``(m+1)``-coaffinity are well defined there too; it is only
the rank-divergence conclusion of Theorem 3.1 that requires ``m >= 2``.)

Combining the two results:

.. math::

    \exists\, m, n, \tau, f \quad\text{such that}\quad
    \tau\text{ is an } (m+1)\text{-coaffination of } G \text{ and }
    f\colon (R_{2m}^n,\sigma)\to(G,\tau)
    \text{ is an admissible graph morphism}

implies that :math:`G` is rank divergent (Theorem 2.6 applied to Theorem 3.1),
and hence that :math:`G` is clique divergent.

The morphism :math:`f` need not be injective: Theorem 2.6 applies to graph
relations, of which graph morphisms are a special case.  Finding no such
certificate within the bounded search performed here proves nothing about
``G``: it is neither evidence of convergence nor evidence that ``G`` lacks a
suitable coaffination.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterator

import networkx as nx

from .clockwork import clockwork_graph
from .coaffinations import CoaffinePair, coaffinations

__all__ = [
    "r_clock",
    "canonical_clockwork_coaffination",
    "clockwork_coaffine_pair",
    "candidate_target_coaffinations",
    "find_pair_morphism",
]


def r_clock(m: int, n: int) -> nx.Graph:
    r"""Return the clockwork graph :math:`R_{2m}^n` of Theorem 3.1.

    The paper defines :math:`R_{2m}^n` only for ``m >= 2``; Theorem 3.1's
    rank-divergence conclusion is only asserted in that range.  ``m == 1``
    is accepted here as a general construction convenience (it is still an
    ``(m + 1)``-coaffine automorphic graph), but must not be treated as a
    rank-divergent Theorem 3.1 source.

    .. rubric:: Parameters

    m : int
        Positive integer; the coaffinity radius of the result is ``m + 1``.
    n : int
        Nonnegative integer controlling the size of each core segment
        (``n + 1`` vertices per segment).

    .. rubric:: Returns

    networkx.Graph
        The clockwork graph :math:`R_{2m}^n`, with ``4 * m`` crown vertices
        numbered first (``0`` to ``4 * m - 1``), followed by
        ``2 * m * (n + 1)`` core vertices.

    .. rubric:: Examples

    >>> from pycliques.clockwork_pairs import r_clock
    >>> r_clock(1, 0).number_of_nodes()
    6
    """
    return clockwork_graph(
        (2 * m) * [n + 1],
        [list(range(n + 1)) for _ in range(2 * m)],
        2,
        [0, 1],
    )


def canonical_clockwork_coaffination(m: int, n: int) -> dict[Hashable, Hashable]:
    r"""Return the canonical :math:`(m+1)`-coaffination of :math:`R_{2m}^n`.

    The automorphism shifts each of the :math:`2m` segments by :math:`m`
    (an involutive half-turn of the cyclic segment structure).  On the
    crown segments (always of size 2) it additionally swaps the two
    positions within the segment; on the core segments (size ``n + 1``) the
    position within the segment is preserved.

    .. rubric:: Parameters

    m : int
        Positive integer parameter of :func:`r_clock`.
    n : int
        Nonnegative integer parameter of :func:`r_clock`.

    .. rubric:: Returns

    dict[Hashable, Hashable]
        The canonical coaffination :math:`\sigma`, verified by
        :func:`clockwork_coaffine_pair` to be an involutive automorphism
        realizing coaffinity radius ``m + 1``.

    .. rubric:: Examples

    >>> from pycliques.clockwork_pairs import canonical_clockwork_coaffination
    >>> sigma = canonical_clockwork_coaffination(1, 0)
    >>> sigma == {0: 3, 1: 2, 2: 1, 3: 0, 4: 5, 5: 4}
    True
    """
    crown_size = 4 * m
    sigma: dict[Hashable, Hashable] = {}
    for v in range(crown_size):
        seg, pos = divmod(v, 2)
        new_seg = (seg + m) % (2 * m)
        sigma[v] = new_seg * 2 + (1 - pos)
    core_size = (2 * m) * (n + 1)
    for v in range(crown_size, crown_size + core_size):
        w = v - crown_size
        seg, pos = divmod(w, n + 1)
        new_seg = (seg + m) % (2 * m)
        sigma[v] = crown_size + new_seg * (n + 1) + pos
    return sigma


def clockwork_coaffine_pair(m: int, n: int) -> CoaffinePair:
    r"""Return the coaffine pair :math:`(R_{2m}^n, \sigma)` of Theorem 3.1.

    Constructs :math:`R_{2m}^n` and its canonical coaffination, and verifies
    (rather than assumes) that :math:`\sigma` is an automorphism, an
    involution, and realizes coaffinity radius ``m + 1``, as guaranteed by
    Theorem 3.1.

    .. rubric:: Parameters

    m : int
        Positive integer; the coaffinity radius is ``m + 1``.
    n : int
        Nonnegative integer core-segment size parameter.

    .. rubric:: Returns

    CoaffinePair
        The pair ``(R_{2m}^n, sigma)``.

    .. rubric:: Raises

    ValueError
        If *m* is not positive or *n* is negative.
    AssertionError
        If the constructed :math:`\sigma` fails to be an automorphism, an
        involution, or an :math:`(m+1)`-coaffination -- this would indicate a
        bug in :func:`canonical_clockwork_coaffination`, not a mathematical
        failure of Theorem 3.1.

    .. rubric:: Examples

    >>> from pycliques.clockwork_pairs import clockwork_coaffine_pair
    >>> pair = clockwork_coaffine_pair(1, 0)
    >>> pair.graph.number_of_nodes()
    6
    """
    if m < 1:
        raise ValueError("m must be a positive integer")
    if n < 0:
        raise ValueError("n must be a nonnegative integer")
    graph = r_clock(m, n)
    sigma = canonical_clockwork_coaffination(m, n)
    _verify_coaffination(graph, sigma, m + 1)
    return CoaffinePair(graph, sigma)


def _verify_coaffination(
    graph: nx.Graph, sigma: dict[Hashable, Hashable], r: int
) -> None:
    """Assert *sigma* is an involutive automorphism of *graph* realizing radius *r*."""
    for u, v in graph.edges():
        if not graph.has_edge(sigma[u], sigma[v]):
            raise AssertionError("sigma is not an automorphism")
    if not all(sigma[sigma[x]] == x for x in graph):
        raise AssertionError("sigma is not an involution")
    distance = dict(nx.all_pairs_shortest_path_length(graph))
    for x in graph:
        if distance[x].get(sigma[x], math.inf) < r:
            raise AssertionError(f"sigma does not realize coaffinity radius {r}")


def candidate_target_coaffinations(
    graph: nx.Graph, radius: int
) -> Iterator[dict[Hashable, Hashable]]:
    r"""Yield involutive :math:`(radius)`-coaffinations of *graph*.

    Reuses :func:`pycliques.coaffinations.coaffinations` to enumerate
    automorphisms satisfying the distance requirement, then keeps only the
    involutions among them (``tau[tau[x]] == x`` for every vertex ``x``),
    per the involutive restriction of this first implementation of the
    Theorem 2.6 classifier rule.

    Vertices in different connected components of a disconnected *graph*
    are treated according to the existing behavior of
    :func:`pycliques.coaffinations.coaffinations`: since a graph
    automorphism can only pair vertices whose components are isomorphic,
    and that helper looks up finite shortest-path lengths, an automorphism
    that would require comparing vertices with no finite path between them
    is not produced as a candidate by this function either.

    .. rubric:: Parameters

    graph : networkx.Graph
        Target graph ``G``.
    radius : int
        Required coaffinity radius (``m + 1`` for a given ``m``).

    .. rubric:: Yields

    dict[Hashable, Hashable]
        Each involutive automorphism ``tau`` of ``graph`` with
        ``d(x, tau(x)) >= radius`` for every vertex ``x``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.clockwork_pairs import candidate_target_coaffinations
    >>> cycle = nx.cycle_graph(4)
    >>> list(candidate_target_coaffinations(cycle, 2))
    [{2: 0, 3: 1, 0: 2, 1: 3}]
    """
    for tau in coaffinations(graph, radius):
        if all(tau[tau[x]] == x for x in graph):
            yield tau


def _orbits(sigma: dict[Hashable, Hashable]) -> list[tuple[Hashable, ...]]:
    """Return the orbits of an involution *sigma* as 1- or 2-element tuples."""
    seen: set[Hashable] = set()
    orbits: list[tuple[Hashable, ...]] = []
    for x in sigma:
        if x in seen:
            continue
        y = sigma[x]
        seen.add(x)
        seen.add(y)
        orbits.append((x,) if y == x else (x, y))
    return orbits


def find_pair_morphism(
    source_pair: CoaffinePair,
    target_pair: CoaffinePair,
) -> dict[Hashable, Hashable] | None:
    r"""Search for an admissible graph morphism of coaffine pairs.

    Finds ``f: source_pair.graph -> target_pair.graph`` such that:

    * ``f`` is a graph morphism: for every edge ``xy`` of the source,
      ``f(x)f(y)`` is an edge of the target;
    * ``f`` is admissible/equivariant: ``f(sigma(x)) == tau(f(x))`` for
      every source vertex ``x``, where ``sigma`` and ``tau`` are the two
      pairs' coaffinations.

    The map ``f`` need not be injective.  The search assigns whole
    coaffination orbits of the source at once: assigning ``f(x) = y`` for
    ``x != sigma(x)`` immediately determines ``f(sigma(x)) = tau(y)``; a
    fixed point ``x == sigma(x)`` may only be sent to a fixed point of
    ``tau``. Orbits are processed in decreasing order of total degree (most
    constrained first), and each partial assignment is checked against
    every edge incident to an already-assigned source vertex before
    recursing.

    .. rubric:: Parameters

    source_pair : CoaffinePair
        The domain pair ``(R, sigma)``, typically from
        :func:`clockwork_coaffine_pair`.
    target_pair : CoaffinePair
        The codomain pair ``(G, tau)``.

    .. rubric:: Returns

    dict[Hashable, Hashable] | None
        A dict representing an admissible morphism ``f``, or ``None`` if
        none exists.

    .. rubric:: Examples

    >>> from pycliques.clockwork_pairs import clockwork_coaffine_pair
    >>> from pycliques.clockwork_pairs import find_pair_morphism
    >>> pair = clockwork_coaffine_pair(1, 0)
    >>> f = find_pair_morphism(pair, pair)
    >>> f is not None
    True
    """
    source = source_pair.graph
    sigma = source_pair.coaffination
    target = target_pair.graph
    tau = target_pair.coaffination

    orbits = _orbits(sigma)
    orbits.sort(key=lambda orb: sum(source.degree(x) for x in orb), reverse=True)

    fixed_targets = [y for y in target if tau[y] == y]
    all_targets = list(target.nodes())

    assignment: dict[Hashable, Hashable] = {}

    def consistent(assign: dict[Hashable, Hashable]) -> bool:
        for x, fx in assign.items():
            for nb in source.neighbors(x):
                if nb in assignment and not target.has_edge(fx, assignment[nb]):
                    return False
                if nb in assign and nb != x and not target.has_edge(fx, assign[nb]):
                    return False
        return True

    def backtrack(i: int) -> dict[Hashable, Hashable] | None:
        if i == len(orbits):
            return dict(assignment)
        orbit = orbits[i]
        candidates = fixed_targets if len(orbit) == 1 else all_targets
        for y in candidates:
            if len(orbit) == 1:
                assign = {orbit[0]: y}
            else:
                x, x2 = orbit
                assign = {x: y, x2: tau[y]}
            if consistent(assign):
                assignment.update(assign)
                found = backtrack(i + 1)
                if found is not None:
                    return found
                for key in assign:
                    del assignment[key]
        return None

    return backtrack(0)
