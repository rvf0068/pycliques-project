"""Bounded strong-product split search with exact coordinate verification."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from itertools import combinations
from math import isqrt

import networkx as nx


@dataclass(frozen=True)
class StrongProductSplit:
    """A bijection onto the vertex set of a nontrivial strong product."""

    coordinates: dict[Hashable, tuple[int, int]]


def _valid_graph(graph: nx.Graph) -> bool:
    return (
        len(graph) > 0
        and not graph.is_directed()
        and not graph.is_multigraph()
        and nx.number_of_selfloops(graph) == 0
        and nx.is_connected(graph)
    )


def strong_product_factors(
    graph: nx.Graph, split: StrongProductSplit
) -> tuple[nx.Graph, nx.Graph]:
    """Reconstruct factors from the coordinates and verify the entire product.

    Every input edge projects to edges or equal vertices in the factors.
    A bijection onto a rectangle and equality of edge counts then prove
    equality with the strong product, without an isomorphism search.
    """
    if not _valid_graph(graph):
        raise ValueError("Require a nonempty connected simple graph")
    coordinates = split.coordinates
    if set(coordinates) != set(graph):
        raise ValueError("Coordinates must cover the input vertices exactly")
    if any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(type(i) is not int or i < 0 for i in pair)
        for pair in coordinates.values()
    ):
        raise ValueError("Coordinates must be pairs of nonnegative integers")
    left = nx.Graph()
    right = nx.Graph()
    left.add_nodes_from(a for a, _ in coordinates.values())
    right.add_nodes_from(b for _, b in coordinates.values())
    if (
        len(left) < 2
        or len(right) < 2
        or len(left) * len(right) != len(graph)
        or len(set(coordinates.values())) != len(graph)
    ):
        raise ValueError("Coordinates must biject onto a nontrivial rectangle")
    for u, v in graph.edges():
        a, b = coordinates[u]
        c, d = coordinates[v]
        if a != c:
            left.add_edge(a, c)
        if b != d:
            right.add_edge(b, d)
    expected = (
        (len(left) + 2 * left.number_of_edges())
        * (len(right) + 2 * right.number_of_edges())
        - len(graph)
    ) // 2
    if expected != graph.number_of_edges():
        raise ValueError("Input graph is not the strong product of its projections")
    return left, right


def verify_strong_product_split(graph: nx.Graph, split: StrongProductSplit) -> bool:
    """Check a proposed coordinate split, independently of the search."""
    try:
        strong_product_factors(graph, split)
    except (ValueError, TypeError, AttributeError, KeyError):
        return False
    return True


def _closed_cartesian_skeleton(graph: nx.Graph) -> nx.Graph:
    """Remove dispensable edges (Hammack and Imrich, 2014, Section 2)."""
    closed = {v: set(graph[v]) | {v} for v in graph}
    skeleton = nx.Graph()
    skeleton.add_nodes_from(graph)
    for x, y in graph.edges():
        a, b = closed[x], closed[y]
        common = a & b
        # Any witness must be adjacent or equal to both endpoints.
        dispensable = any(
            (common < a & closed[z] or a < closed[z] < b)
            and (common < b & closed[z] or b < closed[z] < a)
            for z in common - {x, y}
        )
        if not dispensable:
            skeleton.add_edge(x, y)
    return skeleton


def _square_edge_classes(graph: nx.Graph) -> list[list[tuple[int, int]]]:
    """Generate the standard local square relation on Cartesian edges.

    Incident edges are identified unless they span exactly one chordless
    square. Opposite edges of every chordless square are identified.
    These classes are only search hints, never a product certificate.
    """

    def edge(u: int, v: int) -> tuple[int, int]:
        return min(u, v), max(u, v)

    edges = [edge(u, v) for u, v in graph.edges()]
    union = nx.utils.UnionFind(edges)
    neighbors = {v: set(graph[v]) for v in graph}
    for x in graph:
        for y, z in combinations(sorted(neighbors[x]), 2):
            e, f = edge(x, y), edge(x, z)
            if z in neighbors[y]:
                union.union(e, f)
                continue
            corners = (neighbors[y] & neighbors[z]) - neighbors[x] - {x}
            if len(corners) != 1:
                union.union(e, f)
            for w in corners:
                union.union(e, edge(z, w))
                union.union(f, edge(y, w))
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for e in edges:
        groups[union[e]].append(e)
    return sorted(groups.values(), key=min)


def _components(vertices: range, edges: list[tuple[int, int]]) -> dict[int, int]:
    subgraph = nx.Graph()
    subgraph.add_nodes_from(vertices)
    subgraph.add_edges_from(edges)
    return {
        v: i
        for i, component in enumerate(nx.connected_components(subgraph))
        for v in component
    }


def find_strong_product_split(
    graph: nx.Graph, *, max_vertices: int = 256, max_partitions: int = 128
) -> StrongProductSplit | None:
    """Search for a nontrivial strong-product split of a thin connected graph.

    Thin means distinct closed neighborhoods. Group local square classes of
    the closed Cartesian skeleton into two directions, obtain coordinates
    from components, and verify each proposed rectangle on the original
    graph. The caps bound vertices and partitions, not time. A return of
    None is inconclusive, including on graphs with true twins.
    """
    if type(max_vertices) is not int or max_vertices < 1:
        raise ValueError("max_vertices must be positive")
    if type(max_partitions) is not int or max_partitions < 1:
        raise ValueError("max_partitions must be positive")
    if not _valid_graph(graph):
        raise ValueError("Require a nonempty connected simple graph")
    n = len(graph)
    if n > max_vertices or not any(n % d == 0 for d in range(2, isqrt(n) + 1)):
        return None
    if len({frozenset(graph[v]) | {v} for v in graph}) != n:
        return None
    vertices = tuple(graph)
    indexed = nx.relabel_nodes(graph, {v: i for i, v in enumerate(vertices)})
    skeleton = _closed_cartesian_skeleton(indexed)
    classes = _square_edge_classes(skeleton)
    # Fix the first class on one side to avoid exchanging the two factors.
    full = (1 << len(classes)) - 1
    for mask in range(1, min(full, 2 * max_partitions), 2):
        left_edges = [
            e for i, group in enumerate(classes) if mask & (1 << i) for e in group
        ]
        right_edges = [
            e for i, group in enumerate(classes) if not mask & (1 << i) for e in group
        ]
        a = _components(range(n), right_edges)
        b = _components(range(n), left_edges)
        split = StrongProductSplit({v: (a[i], b[i]) for i, v in enumerate(vertices)})
        if verify_strong_product_split(graph, split):
            return split
    return None
