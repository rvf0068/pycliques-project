"""Graph homomorphisms, their compatibility graph, and the Hom poset.

The Hom poset uses the strict edge-preserving convention of Dochtermann,
including loops when present in the input graphs. It is distinct from the
compatibility graph on ordinary homomorphisms.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import Any

import networkx as nx

from .posets import Poset, order_complex
from .simplex import SimplicialComplex

MultiHomomorphism = frozenset[tuple[Hashable, frozenset[Hashable]]]


class HomPosetLimit(RuntimeError):
    """The configured number of Hom-poset elements was exceeded."""


def _check_graphs(g: nx.Graph, h: nx.Graph) -> None:
    if g.is_directed() or h.is_directed():
        raise ValueError("Hom constructions require undirected graphs")


def _image_assignments(
    g: nx.Graph, h: nx.Graph, *, singletons: bool
) -> Iterator[MultiHomomorphism]:
    """Shared backtracking engine for ordinary and multi-valued maps."""
    _check_graphs(g, h)
    vertices = tuple(g)
    target = tuple(h)
    neighbors = {x: frozenset(h.neighbors(x)) for x in target}
    index = {v: i for i, v in enumerate(vertices)}
    previous: list[list[int]] = [[] for _ in vertices]
    looped = [g.has_edge(v, v) for v in vertices]
    for u, v in g.edges():
        i, j = index[u], index[v]
        if i != j:
            previous[max(i, j)].append(min(i, j))
    current: list[frozenset[Hashable]] = [frozenset()] * len(vertices)

    def choices(pos: int) -> Iterator[frozenset[Hashable]]:
        allowed = set(target)
        for j in previous[pos]:
            for x in current[j]:
                allowed.intersection_update(neighbors[x])
        if looped[pos]:
            # A loop in G requires every ordered pair from this image set
            # to be an edge of H, including diagonal pairs.
            allowed.intersection_update(x for x in target if x in neighbors[x])
        ordered = [x for x in target if x in allowed]
        if singletons:
            for x in ordered:
                yield frozenset({x})
        elif looped[pos]:
            # All cliques of the induced graph on looped target vertices.
            for face in nx.enumerate_all_cliques(h.subgraph(ordered)):
                yield frozenset(face)
        else:
            for size in range(1, len(ordered) + 1):
                for face in itertools.combinations(ordered, size):
                    yield frozenset(face)

    def backtrack(pos: int) -> Iterator[MultiHomomorphism]:
        if pos == len(vertices):
            yield frozenset(zip(vertices, current, strict=True))
            return
        for image in choices(pos):
            current[pos] = image
            yield from backtrack(pos + 1)

    yield from backtrack(0)


def graph_homomorphisms(g: nx.Graph, h: nx.Graph) -> list[dict[Any, Any]]:
    """Enumerate ordinary edge-preserving maps from g to h.

    Each result is a dictionary from domain vertices to target vertices.
    Looped domain vertices must map to looped target vertices. In particular,
    adjacent vertices may coincide under a map only if the image has a loop.
    """
    result: list[dict[Any, Any]] = []
    for eta in _image_assignments(g, h, singletons=True):
        images = dict(eta)
        result.append({v: next(iter(images[v])) for v in g})
    return result


def is_multihomomorphism(
    g: nx.Graph, h: nx.Graph, mapping: Mapping[Hashable, Iterable[Hashable]]
) -> bool:
    """Check nonempty coordinate images and all edge cross-products.

    The mapping must be defined on exactly the vertices of g. Loops in g
    require the full Cartesian square of the corresponding image to consist
    of edges of h.
    """
    _check_graphs(g, h)
    if set(mapping) != set(g):
        return False
    images = {v: frozenset(values) for v, values in mapping.items()}
    target = set(h)
    if any(not values or not values <= target for values in images.values()):
        return False
    return all(
        h.has_edge(x, y)
        for u, v in g.edges()
        for x in images[u]
        for y in images[v]
    )


def iter_multihomomorphisms(g: nx.Graph, h: nx.Graph) -> Iterator[MultiHomomorphism]:
    """Lazily enumerate the multi-valued maps of Dochtermann's Hom poset.

    Labels are frozensets of (domain vertex, frozenset of target vertices)
    pairs; dict(eta) recovers the multivalued mapping. All image sets are
    nonempty, and every source edge maps to a complete cross-product of
    target edges. The empty domain has exactly one map, the empty mapping.
    """
    yield from _image_assignments(g, h, singletons=False)


def hom_poset(
    g: nx.Graph, h: nx.Graph, *, max_elements: int | None = 10_000
) -> Poset:
    """Return Hom(g,h), ordered by pointwise inclusion.

    See A. Dochtermann, Hom complexes and homotopy theory in the category
    of graphs, European J. Combin. 30 (2009), 490-509,
    https://doi.org/10.1016/j.ejc.2008.04.009.

    Each cover adds one target vertex to one coordinate. An exceeded
    max_elements budget raises HomPosetLimit, never returns a partial poset.
    Set max_elements=None to disable the output-size bound.
    """
    if max_elements is not None and (
        not isinstance(max_elements, int)
        or isinstance(max_elements, bool)
        or max_elements < 0
    ):
        raise ValueError("max_elements must be a nonnegative integer or None")
    elements: list[MultiHomomorphism] = []
    for eta in iter_multihomomorphisms(g, h):
        if max_elements is not None and len(elements) >= max_elements:
            raise HomPosetLimit("max_elements exceeded")
        elements.append(eta)

    def covers() -> Iterator[tuple[MultiHomomorphism, MultiHomomorphism]]:
        for eta in elements:
            for v, image in eta:
                if len(image) > 1:
                    rest = eta - {(v, image)}
                    for x in image:
                        yield rest | {(v, image - {x})}, eta

    return Poset(elements, covers())


def hom_order_complex(
    g: nx.Graph, h: nx.Graph, *, max_elements: int | None = 10_000
) -> SimplicialComplex:
    """Order complex of Hom(g,h), the subdivision of the polyhedral Hom complex."""
    return order_complex(hom_poset(g, h, max_elements=max_elements))


def hom_graph(g: nx.Graph, h: nx.Graph) -> nx.Graph:
    """Build the graph ``Hom(G, H)``.

    The vertices of the resulting graph are all graph homomorphisms from
    *g* to *h* (see :func:`graph_homomorphisms`).  Two homomorphisms *f*
    and *phi* are declared adjacent when, for **every** edge ``{x, y}``
    in *g*, both

    * ``f(x)`` is adjacent to ``phi(y)`` in *h*, and
    * ``f(y)`` is adjacent to ``phi(x)`` in *h*.

    The construction is an induced subgraph of the classical *exponential
    graph* ``H^G``.

    Each node of the returned graph is a :class:`frozenset` of
    ``(vertex, image)`` pairs; use ``dict(node)`` to recover the
    homomorphism as a plain dictionary.

    .. rubric:: Parameters

    g : networkx.Graph
        The domain graph.
    h : networkx.Graph
        The codomain graph.

    .. rubric:: Returns

    networkx.Graph
        The hom graph whose nodes are frozen homomorphism mappings.

    .. rubric:: Examples

    A single-vertex domain with no edges means any map is a homomorphism
    and the adjacency condition is vacuously satisfied, so the result is
    the complete graph on the vertices of *H*::

        >>> import networkx as nx
        >>> from pycombtop.hom_complex import hom_graph
        >>> G = nx.complete_graph(1)
        >>> H = nx.complete_graph(2)
        >>> result = hom_graph(G, H)
        >>> result.number_of_nodes()
        2
        >>> result.number_of_edges()
        1

    When *G* is a single edge and *H* is also a single edge (K_2), only
    two homomorphisms exist and they are not adjacent::

        >>> G = nx.path_graph(2)
        >>> H = nx.complete_graph(2)
        >>> result = hom_graph(G, H)
        >>> result.number_of_nodes()
        2
        >>> result.number_of_edges()
        0
    """
    homomorphisms = graph_homomorphisms(g, h)

    result: nx.Graph = nx.Graph()
    result.add_nodes_from(frozenset(f.items()) for f in homomorphisms)

    for f, phi in itertools.combinations(homomorphisms, 2):
        if all(
            h.has_edge(f[u], phi[v]) and h.has_edge(f[v], phi[u]) for u, v in g.edges()
        ):
            result.add_edge(frozenset(f.items()), frozenset(phi.items()))

    return result
