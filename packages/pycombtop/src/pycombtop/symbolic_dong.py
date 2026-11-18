"""Exact symbolic Dong matching for a graph clique complex using a ZDD.

A zero-suppressed decision diagram stores a family of sets, not individual
simplices. Node (v, low, high) denotes low union {s union {v}: s in high}.
Terminal 0 is the empty family; terminal 1 is the family {empty face}.
Zero suppression identifies (v, low, 0) with low.

At pivot v the remaining faces are F^- union v*F^+. Dong's matching removes
F^- intersection F^+ from both halves. The exact remainder is
(F^- \ F^+) union v*(F^+ \ F^-). This is precisely the sequential matching
used by pycombtop.simplex.dong_matching, represented symbolically.
Its acyclicity follows from the sequential matching theorem, not homology.

Caps on the ZDD and graph order return None rather than an unproved verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Hashable, Iterable

import networkx as nx


class _DiagramLimit(Exception):
    """The symbolic search exceeded its node budget."""


@dataclass(frozen=True)
class SymbolicDongCertificate:
    """Exact critical cells in an augmented Dong matching.

    critical maps dimension to multiplicity; dimension -1 denotes an
    unmatched empty face and never certifies a wedge. total_faces includes
    the empty face. critical_masks contains all critical bitmasks when
    there are at most 256 of them, and otherwise is None.
    """

    critical: dict[int, int]
    total_faces: int
    diagram_nodes: int
    critical_masks: tuple[int, ...] | None


class _ZDD:
    """Canonical ordered zero-suppressed diagram of finite set families."""

    def __init__(self, n: int, node_limit: int):
        self.n = n
        self.node_limit = node_limit
        self.variables = [n, n]
        self.low = [0, 0]
        self.high = [0, 0]
        self.unique: dict[tuple[int, int, int], int] = {}

    def node(self, variable: int, low: int, high: int) -> int:
        if high == 0:
            return low
        key = (variable, low, high)
        existing = self.unique.get(key)
        if existing is not None:
            return existing
        if len(self.variables) - 2 >= self.node_limit:
            raise _DiagramLimit
        node = len(self.variables)
        self.variables.append(variable)
        self.low.append(low)
        self.high.append(high)
        self.unique[key] = node
        return node

    @lru_cache(maxsize=500_000)
    def without_vertices(self, root: int, forbidden: int) -> int:
        """Retain only faces that avoid the forbidden vertices."""
        if root < 2 or not forbidden:
            return root
        v = self.variables[root]
        forbidden &= ~((1 << v) - 1)
        if not forbidden:
            return root
        if forbidden & (1 << v):
            return self.without_vertices(self.low[root], forbidden & ~(1 << v))
        return self.node(
            v,
            self.without_vertices(self.low[root], forbidden),
            self.without_vertices(self.high[root], forbidden),
        )

    def clique_faces(self, graph: nx.Graph, vertices: list[Hashable]) -> int:
        """Build the entire face family without enumerating the faces."""
        n = len(vertices)
        neighbors = [0] * n
        index = {v: i for i, v in enumerate(vertices)}
        for u, v in graph.edges:
            i, j = index[u], index[v]
            if i != j:
                neighbors[i] |= 1 << j
                neighbors[j] |= 1 << i
        all_bits = (1 << n) - 1
        root = 1
        for i in range(n - 1, -1, -1):
            higher = all_bits ^ ((1 << (i + 1)) - 1)
            incompatible = higher & ~neighbors[i]
            high = self.without_vertices(root, incompatible)
            root = self.node(i, root, high)
        return root

    @lru_cache(maxsize=500_000)
    def contains_empty(self, root: int) -> bool:
        while root > 1:
            root = self.low[root]
        return root == 1

    @lru_cache(maxsize=500_000)
    def remove_empty(self, root: int) -> int:
        if root < 2:
            return 0
        return self.node(
            self.variables[root], self.remove_empty(self.low[root]), self.high[root]
        )

    @lru_cache(maxsize=500_000)
    def difference(self, a: int, b: int) -> int:
        """Exact difference of two ZDD-encoded set families."""
        if a == 0 or a == b:
            return 0
        if b == 0:
            return a
        if a == 1:
            return 0 if self.contains_empty(b) else 1
        if b == 1:
            return self.remove_empty(a)
        u, v = self.variables[a], self.variables[b]
        if u < v:
            return self.node(u, self.difference(self.low[a], b), self.high[a])
        if v < u:
            return self.difference(a, self.low[b])
        return self.node(
            u,
            self.difference(self.low[a], self.low[b]),
            self.difference(self.high[a], self.high[b]),
        )

    @lru_cache(maxsize=500_000)
    def pivot(self, root: int, vertex: int) -> int:
        """Remove all currently available Dong matching pairs at vertex."""
        if root < 2 or self.variables[root] > vertex:
            return root
        v = self.variables[root]
        if v < vertex:
            return self.node(
                v,
                self.pivot(self.low[root], vertex),
                self.pivot(self.high[root], vertex),
            )
        return self.node(
            v,
            self.difference(self.low[root], self.high[root]),
            self.difference(self.high[root], self.low[root]),
        )

    @lru_cache(maxsize=500_000)
    def face_size_counts(self, root: int) -> tuple[int, ...]:
        if root == 0:
            return ()
        if root == 1:
            return (1,)
        low = list(self.face_size_counts(self.low[root]))
        high = self.face_size_counts(self.high[root])
        low.extend([0] * (max(len(low), len(high) + 1) - len(low)))
        for k, count in enumerate(high):
            low[k + 1] += count
        return tuple(low)

    def masks(self, root: int, prefix: int = 0) -> Iterable[int]:
        if root == 0:
            return
        if root == 1:
            yield prefix
            return
        yield from self.masks(self.low[root], prefix)
        yield from self.masks(self.high[root], prefix | (1 << self.variables[root]))


def symbolic_dong_certificate(
    graph: nx.Graph,
    *,
    vertex_order: Iterable[Hashable] | None = None,
    max_vertices: int = 80,
    max_nodes: int = 250_000,
) -> SymbolicDongCertificate | None:
    """Compute an exact augmented Dong matching without enumerating faces.

    Natural order is used for integer-labelled graphs and the stable key
    of the ordinary Dong implementation for other labels. A supplied
    vertex_order permits replay of any specified order. The certificate
    describes an acyclic matching, but it determines a homotopy wedge only
    when the empty face is paired and all critical faces have one dimension.
    Multiple critical dimensions require additional attaching-map proofs.
    """
    if max_vertices < 0 or max_nodes < 1:
        raise ValueError("max_vertices must be nonnegative and max_nodes positive")
    n = graph.number_of_nodes()
    if n > max_vertices:
        return None
    if vertex_order is None:
        from .simplex import _stable_vertex_key

        if all(isinstance(v, int) for v in graph):
            vertices = sorted(graph.nodes)
        else:
            vertices = sorted(graph.nodes, key=_stable_vertex_key)
    else:
        vertices = list(vertex_order)
        if len(vertices) != n or len(set(vertices)) != n or set(vertices) != set(graph):
            raise ValueError("vertex_order must be a permutation of graph vertices")
    try:
        diagram = _ZDD(n, max_nodes)
        root = diagram.clique_faces(graph, vertices)
        total = sum(diagram.face_size_counts(root))
        for pivot in range(n):
            root = diagram.pivot(root, pivot)
        sizes = diagram.face_size_counts(root)
        critical = {size - 1: count for size, count in enumerate(sizes) if count}
        critical_count = sum(critical.values())
        masks = tuple(diagram.masks(root)) if critical_count <= 256 else None
        return SymbolicDongCertificate(
            critical=critical,
            total_faces=total,
            diagram_nodes=len(diagram.variables) - 2,
            critical_masks=masks,
        )
    except (MemoryError, _DiagramLimit, RecursionError):
        return None
