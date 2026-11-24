"""Incidence-bitset representation of a clique graph and its pared core.

Clique vertices have stable integer IDs; masks encode their base vertices.
No adjacency matrix or edge list of the clique graph is stored.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass

import networkx as nx

from .coaffine_endomorphisms import _validate_graph


def _bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


@dataclass(frozen=True)
class ImplicitCliqueGraph:
    """An induced subgraph of a clique intersection graph.

    Construct with from_graph or from_family. IDs index clique_masks; paring
    preserves those IDs and all original incidence data. Base vertices
    may be arbitrary hashable NetworkX labels.
    """

    base_vertices: tuple[Hashable, ...]
    clique_masks: tuple[int, ...]
    active_mask: int
    _incidence: tuple[int, ...]

    @classmethod
    def from_graph(
        cls, graph: nx.Graph, *, max_cliques: int | None = None
    ) -> ImplicitCliqueGraph | None:
        """Enumerate maximal cliques; return None if the count exceeds a cap."""
        _validate_graph(graph)
        if max_cliques is not None and (
            type(max_cliques) is not int or max_cliques < 0
        ):
            raise ValueError("max_cliques must be a nonnegative integer")
        labels = tuple(graph)
        positions = {v: i for i, v in enumerate(labels)}
        masks: list[int] = []
        for clique in nx.find_cliques(graph):
            if max_cliques is not None and len(masks) >= max_cliques:
                return None
            masks.append(sum(1 << positions[v] for v in clique))
        masks.sort()
        return cls._from_masks(labels, masks)

    @classmethod
    def from_family(cls, family: Iterable[Iterable[Hashable]]) -> ImplicitCliqueGraph:
        """Build the intersection graph of a supplied nonempty-set family.

        Member IDs follow input order, with base labels in first-seen order.
        Distinct nonempty members are required. The caller is responsible
        for supplying the full maximal-clique family when representing K(G).
        No maximal-clique enumeration or explicit edge construction occurs.
        """
        positions: dict[Hashable, int] = {}
        masks: list[int] = []
        seen: set[int] = set()
        for member in family:
            mask = 0
            for v in member:
                mask |= 1 << positions.setdefault(v, len(positions))
            if not mask or mask in seen:
                raise ValueError("family must consist of distinct nonempty sets")
            seen.add(mask)
            masks.append(mask)
        return cls._from_masks(tuple(positions), masks)

    @classmethod
    def _from_masks(
        cls, labels: tuple[Hashable, ...], masks: list[int]
    ) -> ImplicitCliqueGraph:
        # Build incidence via bytearrays instead of repeatedly copying
        # large Python integers once for every clique incidence.
        width = (len(masks) + 7) // 8
        incidence_bytes = [bytearray(width) for _ in labels]
        for i, mask in enumerate(masks):
            for v in _bits(mask):
                incidence_bytes[v][i // 8] |= 1 << (i % 8)
        incidence = tuple(int.from_bytes(row, "little") for row in incidence_bytes)
        return cls(labels, tuple(masks), (1 << len(masks)) - 1, incidence)

    @property
    def incidence_masks(self) -> tuple[int, ...]:
        """Incidence stars in base-label order, including inactive member IDs."""
        return self._incidence

    def __len__(self) -> int:
        return self.active_mask.bit_count()

    def __iter__(self) -> Iterator[int]:
        return _bits(self.active_mask)

    def _check_vertex(self, vertex: int) -> None:
        if (
            type(vertex) is not int
            or not 0 <= vertex < len(self.clique_masks)
            or not self.active_mask & (1 << vertex)
        ):
            raise KeyError(vertex)

    def clique(self, vertex: int) -> frozenset[Hashable]:
        """Decode an active clique vertex into its original base labels."""
        self._check_vertex(vertex)
        return frozenset(
            self.base_vertices[i] for i in _bits(self.clique_masks[vertex])
        )

    def has_edge(self, u: int, v: int) -> bool:
        """Test adjacency by intersection of the two base-clique masks."""
        self._check_vertex(u)
        self._check_vertex(v)
        return u != v and bool(self.clique_masks[u] & self.clique_masks[v])

    def _closed(self, vertex: int, active: int) -> int:
        result = 0
        for v in _bits(self.clique_masks[vertex]):
            result |= self._incidence[v]
        return result & active

    def closed_neighborhood_mask(self, vertex: int) -> int:
        """Return the closed neighborhood as a mask of active clique IDs."""
        self._check_vertex(vertex)
        return self._closed(vertex, self.active_mask)

    def common_neighborhood_mask(self, u: int, v: int) -> int:
        """Return the common open neighborhood of two active vertices."""
        self._check_vertex(u)
        self._check_vertex(v)
        return (
            self._closed(u, self.active_mask)
            & self._closed(v, self.active_mask)
            & ~((1 << u) | (1 << v))
        )

    def universal_vertex(self, subset: int | None = None) -> int | None:
        """Find a universal vertex of an induced subgraph without edges.

        The subset is a mask of active clique IDs, defaulting to all active
        vertices. Return None exactly when no universal vertex exists,
        including the empty induced subgraph. Every rejected candidate has
        a witness in the subset that is not adjacent to it.
        """
        active = self.active_mask if subset is None else subset
        if type(active) is not int or active < 0 or active & ~self.active_mask:
            raise ValueError("subset must be a mask of active clique IDs")
        candidates = active
        while candidates:
            candidate = (candidates & -candidates).bit_length() - 1
            missing = active & ~self._closed(candidate, active)
            if not missing:
                return candidate
            witness = (missing & -missing).bit_length() - 1
            candidates &= self._closed(witness, active)
        return None

    def _dominated(self, vertex: int, active: int) -> int:
        # D(q) is the union of all active cliques disjoint from q.
        # A candidate p is dominated exactly when p misses D(q).
        disjoint = active & ~self._closed(vertex, active)
        forbidden = 0
        for row in self._incidence:
            if row & disjoint:
                forbidden |= row
        return active & ~forbidden & ~(1 << vertex)

    def dominated_vertices_mask(self, vertex: int) -> int:
        """All active vertices other than vertex dominated by vertex."""
        self._check_vertex(vertex)
        return self._dominated(vertex, self.active_mask)

    def number_of_edges(self) -> int:
        """Count edges exactly without constructing them."""
        return sum(self._closed(v, self.active_mask).bit_count() - 1 for v in self) // 2

    def to_networkx(self, *, max_edges: int | None = None) -> nx.Graph | None:
        """Materialize using stable integer IDs; refuse over-budget graphs."""
        if max_edges is not None:
            if type(max_edges) is not int or max_edges < 0:
                raise ValueError("max_edges must be a nonnegative integer")
            if self.number_of_edges() > max_edges:
                return None
        graph = nx.Graph()
        graph.add_nodes_from(self)
        for v in self:
            higher = self._closed(v, self.active_mask) >> (v + 1)
            graph.add_edges_from((v, w + v + 1) for w in _bits(higher))
        return graph

    def _with_active(self, active: int) -> ImplicitCliqueGraph:
        return ImplicitCliqueGraph(
            self.base_vertices, self.clique_masks, active, self._incidence
        )

    def pare(self) -> ImplicitParingResult:
        """Remove dominated vertices until stiff, recording each fold.

        A dominator survives its own batch of removals. Domination is
        preserved under deleting other vertices, so the recorded order
        within each batch is valid.
        """
        active = self.active_mask
        moves: list[tuple[int, int]] = []
        while True:
            before = active
            for q in _bits(before):
                if not active & (1 << q):
                    continue
                removed = self._dominated(q, active)
                moves.extend((p, q) for p in _bits(removed))
                active &= ~removed
            if active == before:
                return ImplicitParingResult(self._with_active(active), tuple(moves))


@dataclass(frozen=True)
class ImplicitParingResult:
    """Pared implicit graph and ordered (removed, dominator) pairs."""

    core: ImplicitCliqueGraph
    moves: tuple[tuple[int, int], ...]


def verify_implicit_paring(
    original: ImplicitCliqueGraph, result: ImplicitParingResult
) -> bool:
    """Replay closed-neighborhood containments and verify final stiffness."""
    active = original.active_mask
    try:
        for p, q in result.moves:
            if (
                type(p) is not int
                or type(q) is not int
                or p == q
                or not 0 <= p < len(original.clique_masks)
                or not 0 <= q < len(original.clique_masks)
                or not active & (1 << p)
                or not active & (1 << q)
            ):
                return False
            if original._closed(p, active) & ~original._closed(q, active):
                return False
            active ^= 1 << p
        if result.core != original._with_active(active):
            return False
        return not any(original._dominated(q, active) for q in _bits(active))
    except (TypeError, ValueError, AttributeError):
        return False
