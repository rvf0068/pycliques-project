"""Lazy raw and pared clique iterates with label and fold provenance."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import cast

import networkx as nx

from .cliques import Clique, _clique_graph_from_index, _cliques_by_vertex
from .dominated import _pare_with_folds


def _sequence_vertex_key(vertex: Hashable) -> tuple:
    """A stable total-order key for ordinary graph labels."""
    if isinstance(vertex, int):
        return (0, vertex)
    if isinstance(vertex, str):
        return (1, vertex)
    if isinstance(vertex, bytes):
        return (2, vertex)
    if isinstance(vertex, tuple):
        return (3, tuple(_sequence_vertex_key(v) for v in vertex))
    if isinstance(vertex, frozenset):
        return (4, tuple(sorted(_sequence_vertex_key(v) for v in vertex)))
    raise ValueError("unsupported label for relabeling; supply vertex_key")


def _sequence_relabel(graph: nx.Graph, vertices: list[Hashable]) -> nx.Graph:
    """Number nodes in the given order, preserving graph attributes."""
    labels = {v: i for i, v in enumerate(vertices)}
    numbered = nx.relabel_nodes(graph, labels, copy=True)
    result = nx.Graph()
    result.graph.update(numbered.graph)
    result.add_nodes_from((i, numbered.nodes[i]) for i in range(len(vertices)))
    result.add_edges_from(
        (u, v, numbered[u][v])
        for u in range(len(vertices))
        for v in sorted(numbered[u])
        if u <= v
    )
    return result


class CliqueSequence:
    """Lazily cache clique iterates, with optional paring and stable labels.

    By default the supplied graph is level zero, assumed already pared, and
    later levels are completely pared as in the original API. Set
    ``pare_initial=True`` to pare the starting graph as well. Set
    ``pare=False`` for the actual clique iterates (incompatible with
    ``pare_initial=True``).

    With ``relabel=True``, the input is numbered in deterministic label order,
    raw clique vertices in lexicographic order of their preceding integer
    vertices, and survivors in raw integer order. This is not graph
    canonization. A custom ``vertex_key`` may order other input labels; its
    keys must be distinct, stable and mutually comparable. The default
    supports integers, strings, bytes, and nested tuples and frozensets.

    The clique count and edge limits apply to each raw positive iterate,
    before paring. Requesting a clique family alone does not construct any
    edges at that level. Returned graphs should be treated as read-only.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> seq = CliqueSequence(nx.path_graph(4), relabel=True, pare_initial=True)
    >>> seq[0].order()
    1
    >>> len(seq.paring_certificate(0))
    3
    >>> seq = CliqueSequence(nx.octahedral_graph(), pare=False, relabel=True)
    >>> len(seq.clique_family(1))
    8
    >>> seq.graph_count
    1
    >>> seq[1].order()
    8
    """

    def __init__(
        self,
        graph: nx.Graph,
        bound: int | float = 30,
        *,
        pare: bool = True,
        relabel: bool = False,
        max_edges: int | None = None,
        pare_initial: bool = False,
        vertex_key: Callable[[Hashable], tuple] | None = None,
    ) -> None:
        if bound < 0 or (max_edges is not None and max_edges < 0):
            raise ValueError("bounds must be nonnegative")
        if pare_initial and not pare:
            raise ValueError("pare_initial requires pare=True")
        if vertex_key is not None and not relabel:
            raise ValueError("vertex_key requires relabel=True")
        self._bound = bound
        self._pare = pare
        self._relabel = relabel
        self._max_edges = max_edges
        self._exhausted = False
        self._families: dict[int, tuple[Clique, ...]] = {}
        self._raw_graphs: dict[int, nx.Graph] = {}
        self._raw_maps: dict[int, dict[Hashable, Hashable]] = {}
        self._maps: dict[int, dict[Hashable, Hashable]] = {}
        self._certificates: dict[int, tuple[tuple[Hashable, Hashable], ...]] = {}

        vertices: list[Hashable] = list(graph)
        if relabel:
            key = vertex_key or _sequence_vertex_key
            vertices.sort(key=key)
            keys = [key(v) for v in vertices]
            if any(a == b for a, b in zip(keys, keys[1:])):
                raise ValueError("vertex_key must distinguish all input vertices")
            raw = _sequence_relabel(graph, vertices)
            raw_map: dict[Hashable, Hashable] = dict(enumerate(vertices))
        else:
            raw = graph
            raw_map = {v: v for v in vertices}
        self._raw_graphs[0] = raw
        self._raw_maps[0] = raw_map
        self._graphs: list[nx.Graph] = []
        self._append(0, raw, pare=pare_initial)

    @property
    def graph_count(self) -> int:
        """Return the number of cached output iterates, including level zero."""
        return len(self._graphs)

    @property
    def exhausted(self) -> bool:
        """Whether a clique-count or edge bound prevented a requested level."""
        return self._exhausted

    def _append(self, i: int, raw: nx.Graph, *, pare: bool) -> None:
        if pare:
            graph, moves = _pare_with_folds(raw, deterministic=self._relabel)
        else:
            graph, moves = raw, ()
        survivors: list[Hashable] = list(graph)
        raw_map = self._raw_maps[i]
        if self._relabel:
            survivors.sort(key=lambda v: cast(int, v))
            self._maps[i] = {j: raw_map[v] for j, v in enumerate(survivors)}
            if moves:
                graph = _sequence_relabel(graph, survivors)
        else:
            self._maps[i] = {v: raw_map[v] for v in survivors}
        self._certificates[i] = moves
        self._graphs.append(graph)

    def clique_family(self, i: int) -> tuple[Clique, ...] | None:
        """Return the full maximal-clique family of ``seq[i-1]``.

        For positive i only. With relabeling, its tuple indices are the raw
        integer labels at level i, before paring. The final graph's labels
        may differ. No edges of the requested raw level are constructed.
        Return None if a preceding graph or this clique family exceeds a
        bound.
        """
        if i <= 0:
            raise IndexError("clique_family requires a positive iterate")
        if i in self._families:
            return self._families[i]
        previous = self[i - 1]
        if previous is None or self._exhausted:
            return None
        indexed = _cliques_by_vertex(previous, self._bound)
        if indexed is None:
            self._exhausted = True
            return None
        family = indexed[0]
        if self._relabel:
            family.sort(key=lambda q: tuple(sorted(q)))
        result = tuple(family)
        self._families[i] = result
        self._raw_maps[i] = (
            dict(enumerate(result)) if self._relabel else {q: q for q in result}
        )
        return result

    def _build_raw(self, i: int) -> nx.Graph | None:
        if i in self._raw_graphs:
            return self._raw_graphs[i]
        family = self.clique_family(i)
        if family is None:
            return None
        # Reuse the family, including when it was obtained before edge
        # construction. Build edges exactly as clique_graph does.
        previous = self._graphs[i - 1]
        incidence: dict[Hashable, set[Clique]] = {v: set() for v in previous}
        for q in family:
            for v in q:
                incidence[v].add(q)
        raw = _clique_graph_from_index(family, incidence, self._max_edges)
        if raw is None:
            self._exhausted = True
            return None
        if self._relabel:
            raw = _sequence_relabel(raw, list(family))
        return raw

    def raw_iterate(self, i: int) -> nx.Graph | None:
        """Return and cache the raw level before any paring.

        At zero this is the supplied graph, numbered if relabeling is enabled.
        At positive i it is K(seq[i-1]), not necessarily K^i(G).
        """
        if i < 0:
            raise IndexError("iterate must be nonnegative")
        if i < len(self._graphs) and not self._certificates[i]:
            return self._graphs[i]
        raw = self._build_raw(i)
        if raw is not None:
            self._raw_graphs[i] = raw
        return raw

    def raw_label_map(self, i: int) -> dict[Hashable, Hashable] | None:
        """Map raw labels to preceding cliques (input labels at level zero).

        This needs only the clique family, not the raw edges.
        """
        if i < 0:
            raise IndexError("iterate must be nonnegative")
        if i > 0 and self.clique_family(i) is None:
            return None
        return dict(self._raw_maps[i])

    def label_map(self, i: int) -> dict[Hashable, Hashable] | None:
        """Map surviving output labels to their original cliques or labels."""
        if self[i] is None:
            return None
        return dict(self._maps[i])

    def paring_certificate(
        self, i: int
    ) -> tuple[tuple[Hashable, Hashable], ...] | None:
        """Return ordered (deleted vertex, dominator) pairs in raw labels.

        An empty tuple means no deletion; None means the level is unavailable.
        Replay on raw_iterate(i), before compacting the surviving labels.
        """
        if self[i] is None:
            return None
        return self._certificates[i]

    def __getitem__(self, i: int) -> nx.Graph | None:
        """Return the requested output level, or None if a bound was exceeded."""
        if i < 0:
            raise IndexError("iterate must be nonnegative")
        while len(self._graphs) <= i and not self._exhausted:
            level = len(self._graphs)
            raw = self._build_raw(level)
            if raw is None:
                return None
            self._append(level, raw, pare=self._pare)
        if i < len(self._graphs):
            return self._graphs[i]
        return None
