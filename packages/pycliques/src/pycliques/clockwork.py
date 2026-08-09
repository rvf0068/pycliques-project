r"""Construction and recognition of clockwork graphs.

A clockwork graph :math:`G = B \oplus C` is the segmented sum of:

* :math:`B` -- a crown graph (cyclic segmentation; consecutive segments
  joined by perfect matchings; each segment is a clique of size
  :math:`\ge 2`)
* :math:`C` -- a core graph (cyclic segmentation; each segment is a
  clique; adjacency across consecutive segments obeys a strict linear
  order satisfying conditions C1 and C2)

References
----------
Larrion, Neumann-Lara, Pizana (2004).
"Clique divergent clockwork graphs and partial orders."
*Discrete Applied Mathematics* 141, 195--207.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable

import networkx as nx

from .dominated import find_dominated_vertex

# -------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------


def _nbrs(G: nx.Graph, v: int) -> set[int]:
    """Return the open neighborhood of *v*."""
    return set(G.neighbors(v))


def _nbrs_of_set(G: nx.Graph, vset: Iterable[int]) -> set[int]:
    """Union of open neighborhoods over all vertices in *vset*."""
    result: set[int] = set()
    for v in vset:
        result |= _nbrs(G, v)
    return result


def _seg_idx(v: int, segs: list[set[int]]) -> int:
    """Return the index of the segment containing *v*, or -1."""
    for i, s in enumerate(segs):
        if v in s:
            return i
    return -1


def _find_cyclic_order(seg_adj: dict[int, set[int]], s: int) -> list[int] | None:
    """Return a Hamiltonian cycle on *s* nodes, or ``None``."""
    if s < 3:
        return None
    start = min(seg_adj)
    nbrs = list(seg_adj[start])
    if len(nbrs) != 2:
        return None
    for first in nbrs:
        order = [start, first]
        prev, cur = start, first
        ok = True
        for _ in range(s - 2):
            nxt = [n for n in seg_adj[cur] if n != prev]
            if len(nxt) != 1 or nxt[0] in order:
                ok = False
                break
            order.append(nxt[0])
            prev, cur = cur, nxt[0]
        if ok and len(order) == s and start in seg_adj[cur]:
            return order
    return None


def _build_linear_order(
    vertices: list[int],
    po1: Callable[[int, int], bool],
    po2: Callable[[int, int], bool],
) -> list[int] | None:
    """Build a strict linear order compatible with both preorders.

    Returns ``[v_min, ..., v_max]`` or ``None``.
    """
    order: list[int] = []
    for v in vertices:
        placed = False
        for pos in range(len(order) + 1):
            if all(po1(u, v) and po2(u, v) for u in order[:pos]) and all(
                po1(v, u) and po2(v, u) for u in order[pos:]
            ):
                order.insert(pos, v)
                placed = True
                break
        if not placed:
            return None
    return order


# -------------------------------------------------------------------
# Low-level graph predicates
# -------------------------------------------------------------------


def has_induced_4cycle(G: nx.Graph, vertices: Iterable[int]) -> bool:
    r"""Return whether the subgraph on *vertices* contains :math:`C_4`.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.clockwork import has_induced_4cycle
    >>> has_induced_4cycle(nx.cycle_graph(4), [0, 1, 2, 3])
    True
    >>> has_induced_4cycle(nx.complete_graph(4), [0, 1, 2, 3])
    False
    """
    vlist = list(vertices)
    if len(vlist) < 4:
        return False
    sub = G.subgraph(vlist)
    for quad in itertools.combinations(vlist, 4):
        sg = sub.subgraph(quad)
        if sg.number_of_edges() == 4 and all(d == 2 for _, d in sg.degree()):
            return True
    return False


def is_complete_graph(G: nx.Graph, vertices: Iterable[int]) -> bool:
    """Return whether the subgraph on *vertices* is a clique.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.clockwork import is_complete_graph
    >>> is_complete_graph(nx.complete_graph(4), [0, 1, 2])
    True
    >>> is_complete_graph(nx.path_graph(4), [0, 1, 2])
    False
    """
    vlist = list(vertices)
    k = len(vlist)
    return k < 2 or G.subgraph(vlist).number_of_edges() == (k * (k - 1) // 2)


# -------------------------------------------------------------------
# Graph constructors
# -------------------------------------------------------------------


def core(
    segments: list[int], neig_segments: list[list[int]]
) -> tuple[nx.Graph, list[list[int]]]:
    """Return the core graph for the given segment sizes.

    .. rubric:: Parameters

    segments : list[int]
        Sizes of the *s* core segments.  Vertices are numbered
        ``0 .. sum(segments) - 1`` with segment *i* occupying
        ``sum(segments[:i]) .. sum(segments[:i+1]) - 1``.
    neig_segments : list[list[int]]
        ``neig_segments[i][j] = k`` means vertex *j* of segment
        *i* is adjacent to vertices ``0..k-1`` of segment
        ``(i+1) % s``.

    .. rubric:: Returns

    tuple[nx.Graph, list[list[int]]]
        The core graph and its segment list.

    .. rubric:: Examples

    >>> from pycliques.clockwork import core
    >>> G, segs = core([1, 1, 1], [[1], [0], [0]])
    >>> G.number_of_nodes()
    3
    >>> segs
    [[0], [1], [2]]
    """
    core_graph = nx.Graph()
    core_graph.add_nodes_from(range(sum(segments)))
    segment_list: list[list[int]] = []
    for seg in range(len(segments)):
        segment_list.append(
            list(
                range(
                    sum(segments[:seg]),
                    sum(segments[:(seg + 1)]),
                )
            )
        )
        for i in range(segments[seg]):
            curr_vertex = sum(segments[:seg]) + i
            if seg < len(segments) - 1:
                start_next = sum(segments[:(seg + 1)])
            else:
                start_next = 0
            for j in range(neig_segments[seg][i]):
                core_graph.add_edge(curr_vertex, start_next + j)
    for segment in segment_list:
        core_graph.add_edges_from(itertools.combinations(segment, 2))
    return core_graph, segment_list


def crown(
    num_segments: int,
    size_segments: int,
    permutation: list[int],
) -> tuple[nx.Graph, list[list[int]]]:
    """Return the crown graph with uniform segment size.

    Between consecutive segments the matching is the identity.
    The wrap-around matching (last segment to segment 0) is given
    by *permutation*.

    .. rubric:: Parameters

    num_segments : int
        Number of crown segments.
    size_segments : int
        Uniform size of every crown segment.
    permutation : list[int]
        Permutation of ``range(size_segments)`` for the
        wrap-around matching.

    .. rubric:: Returns

    tuple[nx.Graph, list[list[int]]]
        The crown graph and its segment list.

    .. rubric:: Examples

    >>> from pycliques.clockwork import crown
    >>> G, segs = crown(3, 2, [1, 0])
    >>> G.number_of_nodes()
    6
    >>> len(segs)
    3
    """
    crown_graph = nx.Graph()
    n = num_segments * size_segments
    crown_graph.add_nodes_from(range(n))
    for seg in range(num_segments):
        for i in range(size_segments):
            curr = size_segments * seg + i
            if seg < num_segments - 1:
                end = size_segments * (seg + 1) + i
            else:
                end = permutation[i]
            crown_graph.add_edge(curr, end)
    segment_list = [
        list(range(size_segments * i, size_segments * (i + 1)))
        for i in range(num_segments)
    ]
    for segment in segment_list:
        crown_graph.add_edges_from(itertools.combinations(segment, 2))
    return crown_graph, segment_list


def crown_general(
    size_segments: int, crown_matchings: list[list[int]]
) -> tuple[nx.Graph, list[list[int]]]:
    """Return a crown graph with arbitrary per-segment matchings.

    .. rubric:: Parameters

    size_segments : int
        Uniform size of every crown segment.
    crown_matchings : list[list[int]]
        ``crown_matchings[i]`` is a permutation of
        ``range(size_segments)`` encoding the matching from
        segment *i* to segment ``(i+1) % s``.

    .. rubric:: Returns

    tuple[nx.Graph, list[list[int]]]
        The crown graph and its segment list.

    .. rubric:: Examples

    >>> from pycliques.clockwork import crown_general
    >>> G, segs = crown_general(2, [[0, 1], [0, 1], [1, 0]])
    >>> G.number_of_nodes()
    6
    """
    num_segments = len(crown_matchings)
    crown_graph = nx.Graph()
    n = num_segments * size_segments
    crown_graph.add_nodes_from(range(n))
    segment_list = [
        list(range(size_segments * i, size_segments * (i + 1)))
        for i in range(num_segments)
    ]
    for seg in range(num_segments):
        bi = segment_list[seg]
        bj = segment_list[(seg + 1) % num_segments]
        perm = crown_matchings[seg]
        for j in range(size_segments):
            crown_graph.add_edge(bi[j], bj[perm[j]])
        crown_graph.add_edges_from(itertools.combinations(bi, 2))
    return crown_graph, segment_list


def segmented_sum(
    segmented_b: tuple[nx.Graph, list[list[int]]],
    segmented_c: tuple[nx.Graph, list[list[int]]],
) -> nx.Graph:
    r"""Return the segmented sum :math:`B \oplus C`.

    In :math:`G = B \oplus C` every vertex of :math:`B_i` and
    :math:`B_{i+1}` is adjacent to every vertex of :math:`C_i`.

    .. rubric:: Parameters

    segmented_b : tuple[nx.Graph, list[list[int]]]
        Crown graph and its segment list.
    segmented_c : tuple[nx.Graph, list[list[int]]]
        Core graph and its segment list.

    .. rubric:: Returns

    nx.Graph
        The clockwork graph :math:`G = B \oplus C`.
    """
    graph_b, segments_b = segmented_b
    graph_c, segments_c = segmented_c
    g = nx.disjoint_union(graph_b, graph_c)
    offset = graph_b.order()
    new_c = [[x + offset for x in seg] for seg in segments_c]
    for i in range(len(segments_c)):
        g.add_edges_from((u, v) for u in segments_b[i] for v in new_c[i])
        j = (i + 1) % len(segments_c)
        g.add_edges_from((u, v) for u in segments_b[j] for v in new_c[i])
    return g


def clockwork_graph(
    segments: list[int],
    neig_segments: list[list[int]],
    size_seg_crown: int,
    permutation: list[int],
) -> nx.Graph:
    """Return the clockwork graph for the given parameters.

    Convenience wrapper that calls :func:`core`, :func:`crown`,
    and :func:`segmented_sum`.  The crown uses identity matchings
    except for the wrap-around given by *permutation*.

    .. rubric:: Parameters

    segments : list[int]
        Core segment sizes.
    neig_segments : list[list[int]]
        Adjacency-count table (see :func:`core`).
    size_seg_crown : int
        Uniform size of each crown segment.
    permutation : list[int]
        Wrap-around permutation (see :func:`crown`).

    .. rubric:: Returns

    nx.Graph
        The clockwork graph.

    .. rubric:: Examples

    >>> from pycliques.clockwork import clockwork_graph
    >>> G = clockwork_graph(
    ...     [1, 1, 1], [[1], [0], [0]], 2, [1, 0]
    ... )
    >>> G.number_of_nodes()
    9
    """
    the_core = core(segments, neig_segments)
    the_crown = crown(len(segments), size_seg_crown, permutation)
    return segmented_sum(the_crown, the_core)


# -------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------


class ClockworkStructure:
    r"""Decomposition :math:`G = B \oplus C` with structural data.

    .. rubric:: Attributes

    G : nx.Graph
        The original graph.
    C_nodes : set[int]
        Core vertices.
    B_nodes : set[int]
        Crown vertices.
    s : int
        Number of segments.
    B_segments : list[set[int]]
        ``B_segments[i]`` -- crown segment *i*.
    C_segments : list[set[int]]
        ``C_segments[i]`` -- core segment *i*.
    core_orders : list[list[int]]
        ``core_orders[i]`` -- vertices of :math:`C_i` ordered
        smallest to largest.
    covered_vertices : set[int]
        Covered vertices (Theorem 3.6).
    good_segments : list[int]
        Indices of good segments.
    """

    def __init__(
        self,
        G: nx.Graph,
        C_nodes: set[int],
        B_nodes: set[int],
        s: int,
        B_segments: list[set[int]],
        C_segments: list[set[int]],
        core_orders: list[list[int]],
        covered_vertices: set[int],
        good_segments: list[int],
    ) -> None:
        self.G = G
        self.C_nodes = C_nodes
        self.B_nodes = B_nodes
        self.s = s
        self.B_segments = B_segments
        self.C_segments = C_segments
        self.core_orders = core_orders
        self.covered_vertices = covered_vertices
        self.good_segments = good_segments

    def __repr__(self) -> str:
        lines = [
            "ClockworkStructure(",
            f"  s (segments)   = {self.s}",
            f"  Crown B        = {sorted(self.B_nodes)}",
            f"  Core  C        = {sorted(self.C_nodes)}",
        ]
        for i in range(self.s):
            b = sorted(self.B_segments[i])
            c = sorted(self.C_segments[i])
            o = self.core_orders[i]
            lines.append(f"  Segment {i}: B={b}, C={c}, core order = {o}")
        lines.append(f"  Good segments  = {self.good_segments}")
        lines.append(f"  Covered verts  = {sorted(self.covered_vertices)}")
        lines.append(")")
        return "\n".join(lines)


# -------------------------------------------------------------------
# ClockworkGraph class
# -------------------------------------------------------------------


class ClockworkGraph:
    r"""A clockwork graph :math:`G = B \oplus C` with parameters.

    Couples the :func:`clockwork_graph` construction API with the
    recognition algorithm :func:`recognize_clockwork`:

    * Build directly from parameters.
    * Convert a recognised :class:`ClockworkStructure` via
      :meth:`from_structure`.
    * ``repr(cg)`` yields an ``eval``-able reconstruction.

    .. rubric:: Parameters

    core_segments : list[int]
        Sizes of the *s* core segments.
    neig_segments : list[list[int]]
        Adjacency-count table (see :func:`core`).
    crown_size : int
        Uniform size of every crown segment.
    crown_matchings : list[list[int]]
        ``crown_matchings[i]`` is a permutation of
        ``range(crown_size)`` encoding the matching from
        :math:`B_i` to :math:`B_{(i+1) \bmod s}`.

    .. rubric:: Examples

    >>> from pycliques.clockwork import ClockworkGraph
    >>> cg = ClockworkGraph(
    ...     [1, 1, 1],
    ...     [[1], [0], [0]],
    ...     2,
    ...     [[0, 1], [0, 1], [1, 0]],
    ... )
    >>> cg.graph.number_of_nodes()
    9
    >>> cg.s
    3
    """

    def __init__(
        self,
        core_segments: list[int],
        neig_segments: list[list[int]],
        crown_size: int,
        crown_matchings: list[list[int]],
    ) -> None:
        if len(core_segments) != len(neig_segments):
            raise ValueError(
                f"core_segments has {len(core_segments)} "
                f"entries but neig_segments has "
                f"{len(neig_segments)}."
            )
        if len(core_segments) != len(crown_matchings):
            raise ValueError(
                f"core_segments has {len(core_segments)} "
                f"entries but crown_matchings has "
                f"{len(crown_matchings)}."
            )
        self.core_segments = list(core_segments)
        self.neig_segments = [list(r) for r in neig_segments]
        self.crown_size = crown_size
        self.crown_matchings = [list(p) for p in crown_matchings]
        self.s = len(core_segments)
        self.graph = self._build()

    def _build(self) -> nx.Graph:
        """Build the underlying NetworkX graph."""
        the_core = core(self.core_segments, self.neig_segments)
        the_crown = crown_general(self.crown_size, self.crown_matchings)
        return segmented_sum(the_crown, the_core)

    @classmethod
    def from_structure(cls, structure: ClockworkStructure) -> ClockworkGraph:
        """Construct from a recognised :class:`ClockworkStructure`.

        .. rubric:: Parameters

        structure : ClockworkStructure
            Output of :func:`recognize_clockwork` on success.

        .. rubric:: Returns

        ClockworkGraph

        .. rubric:: Raises

        ValueError
            If crown segments have non-uniform size.
        """
        cs = structure
        G = cs.G
        s = cs.s

        core_segs = [len(cs.C_segments[i]) for i in range(s)]

        neig_segs: list[list[int]] = []
        for i in range(s):
            Ci_next = cs.C_segments[(i + 1) % s]
            order_next = cs.core_orders[(i + 1) % s]
            row: list[int] = []
            for v in cs.core_orders[i]:
                nbrs_in_next = set(G.neighbors(v)) & Ci_next
                k = len(nbrs_in_next)
                if set(order_next[:k]) != nbrs_in_next:
                    raise ValueError(
                        f"Vertex {v} in C_{i} has "
                        f"C_{{i+1}}-neighbours "
                        f"{sorted(nbrs_in_next)} that do "
                        f"not form a prefix of the "
                        f"C_{{i+1}} order {order_next}."
                    )
                row.append(k)
            neig_segs.append(row)

        sizes = [len(cs.B_segments[i]) for i in range(s)]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"Crown segments have non-uniform sizes "
                f"{sizes}; ClockworkGraph requires a "
                f"uniform crown segment size."
            )
        crown_sz = sizes[0]

        matchings: list[list[int]] = []
        for i in range(s):
            bi = sorted(cs.B_segments[i])
            bj = sorted(cs.B_segments[(i + 1) % s])
            bj_set = set(bj)
            perm: list[int] = []
            for u in bi:
                nbrs = [v for v in G.neighbors(u) if v in bj_set]
                if len(nbrs) != 1:
                    raise ValueError(
                        f"Crown vertex {u} has "
                        f"{len(nbrs)} neighbour(s) in "
                        f"B_{{i+1}} = {bj}; expected 1."
                    )
                perm.append(bj.index(nbrs[0]))
            matchings.append(perm)

        return cls(core_segs, neig_segs, crown_sz, matchings)

    def __repr__(self) -> str:
        """Return an ``eval``-able representation.

        .. rubric:: Examples

        >>> from pycliques.clockwork import ClockworkGraph
        >>> cg = ClockworkGraph(
        ...     [1, 1, 1],
        ...     [[1], [0], [0]],
        ...     2,
        ...     [[0, 1], [0, 1], [1, 0]],
        ... )
        >>> cg2 = eval(repr(cg))
        >>> cg == cg2
        True
        """
        return (
            f"ClockworkGraph("
            f"core_segments={self.core_segments!r}, "
            f"neig_segments={self.neig_segments!r}, "
            f"crown_size={self.crown_size!r}, "
            f"crown_matchings={self.crown_matchings!r})"
        )

    def __str__(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            "ClockworkGraph {",
            f"  s (segments)    = {self.s}",
            f"  core_segments   = {self.core_segments}",
            f"  neig_segments   = {self.neig_segments}",
            f"  crown_size      = {self.crown_size}",
            f"  crown_matchings = {self.crown_matchings}",
            f"  |V(G)|          = {self.graph.order()}",
            f"  |E(G)|          = {self.graph.number_of_edges()}",
            "}",
        ]
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        """Isomorphism-based equality."""
        if not isinstance(other, ClockworkGraph):
            return NotImplemented
        return bool(nx.is_isomorphic(self.graph, other.graph))


# -------------------------------------------------------------------
# Recognition algorithm: Theorem 3.5
# -------------------------------------------------------------------


def recognize_clockwork(
    G: nx.Graph,
) -> tuple[bool, ClockworkStructure | str]:
    r"""Decide whether *G* is a clockwork graph.

    Follows Theorem 3.5 of Larrion *et al.* (2004), using
    Theorem 3.4 as the key structural test.

    .. note::

       Reliably handles :math:`s \ge 4`.  For :math:`s = 3` the
       algorithm may return ``(False, ...)`` on a valid clockwork
       graph due to Theorem 3.4's extra triangle condition.

    .. rubric:: Parameters

    G : nx.Graph
        A finite simple graph.

    .. rubric:: Returns

    tuple[bool, ClockworkStructure | str]
        ``(True, structure)`` if *G* is clockwork;
        ``(False, reason)`` otherwise.

    .. rubric:: Examples

    >>> from pycliques.clockwork import (
    ...     clockwork_graph, recognize_clockwork,
    ... )
    >>> G = clockwork_graph(
    ...     [1, 1, 1], [[1], [0], [0]], 2, [1, 0]
    ... )
    >>> ok, result = recognize_clockwork(G)
    >>> ok
    True
    >>> result.s
    3
    """
    if len(G) == 0:
        return False, "Empty graph."
    if not nx.is_connected(G):
        return False, "Graph is not connected."

    # Step 1. Core C vs crown B via Theorem 3.4.
    C_nodes: set[int] = {v for v in G if has_induced_4cycle(G, G.neighbors(v))}
    B_nodes = set(G.nodes()) - C_nodes

    if not C_nodes:
        return False, (
            "Core C is empty: no vertex whose neighbourhood contains an induced C4."
        )
    if not B_nodes:
        return False, (
            "Crown B is empty: every vertex's neighbourhood contains an induced C4."
        )
    if not nx.is_connected(G.subgraph(B_nodes)):
        return False, "Crown subgraph B is disconnected."

    # Step 2. Partition B into crown segments.
    b_class: dict[frozenset[int], set[int]] = {}
    for v in B_nodes:
        key = frozenset(_nbrs(G, v) & C_nodes)
        b_class.setdefault(key, set()).add(v)

    B_segs_raw = list(b_class.values())
    s = len(B_segs_raw)

    if s < 3:
        return False, (f"Only {s} crown segment(s) -- need s >= 3.")

    for i, bs in enumerate(B_segs_raw):
        if len(bs) < 2:
            return False, (
                f"Crown segment {i} has {len(bs)} vertex -- violates B1 (need >= 2)."
            )
        if not is_complete_graph(G, bs):
            return False, (f"Crown segment {i} is not a clique.")

    seg_of: dict[int, int] = {}
    for i, bs in enumerate(B_segs_raw):
        for v in bs:
            seg_of[v] = i

    seg_adj: dict[int, set[int]] = {i: set() for i in range(s)}
    for v in B_nodes:
        i = seg_of[v]
        for u in G.neighbors(v):
            if u in B_nodes and seg_of[u] != i:
                seg_adj[i].add(seg_of[u])

    for i in range(s):
        if len(seg_adj[i]) != 2:
            return False, (
                f"Crown segment {i} is adjacent to "
                f"{len(seg_adj[i])} other segment(s); "
                f"expected 2."
            )

    cyc = _find_cyclic_order(seg_adj, s)
    if cyc is None:
        return False, ("Crown segments do not form a simple Hamiltonian cycle.")

    B_segs = [B_segs_raw[i] for i in cyc]

    # Verify B2: perfect matchings.
    for i in range(s):
        bi, bj = B_segs[i], B_segs[(i + 1) % s]
        l_deg: dict[int, int] = {v: 0 for v in bi}
        r_deg: dict[int, int] = {v: 0 for v in bj}
        for u in bi:
            for w in bj:
                if G.has_edge(u, w):
                    l_deg[u] += 1
                    r_deg[w] += 1
        if not (
            all(d == 1 for d in l_deg.values()) and all(d == 1 for d in r_deg.values())
        ):
            return False, (
                f"Edges between crown segments {i} and "
                f"{(i + 1) % s} are not a perfect "
                f"matching -- violates B2."
            )

    # Step 3. Core segments.
    C_segs: list[set[int]] = []
    for i in range(s):
        nbrs_bi = _nbrs_of_set(G, B_segs[i]) & C_nodes
        nbrs_bj = _nbrs_of_set(G, B_segs[(i + 1) % s]) & C_nodes
        C_segs.append(nbrs_bi & nbrs_bj)

    for i, Ci in enumerate(C_segs):
        if not Ci:
            return False, f"Core segment C_{i} is empty."
        if not is_complete_graph(G, Ci):
            return False, (f"Core segment C_{i} is not a clique.")

    seen: set[int] = set()
    for i, Ci in enumerate(C_segs):
        overlap = Ci & seen
        if overlap:
            return False, (
                f"Core segment C_{i} overlaps earlier segments on {overlap}."
            )
        seen |= Ci
    if seen != C_nodes:
        return False, (f"Core vertices {C_nodes - seen} are not in any C_i.")

    for i, Ci in enumerate(C_segs):
        allowed = C_segs[(i - 1) % s] | Ci | C_segs[(i + 1) % s]
        for v in Ci:
            bad = (_nbrs(G, v) & C_nodes) - allowed
            if bad:
                return False, (
                    f"Core vertex {v} in C_{i} has "
                    f"core-neighbours {bad} outside "
                    f"adjacent segments."
                )

    for a, b, c in itertools.combinations(C_nodes, 3):
        if G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(a, c):
            idxs = {_seg_idx(v, C_segs) for v in (a, b, c)}
            if len(idxs) == 3:
                return False, (
                    f"Triangle ({a},{b},{c}) spans three distinct core segments."
                )

    # Step 4. Core orders (C1 and C2 conditions).
    core_orders: list[list[int]] = []
    for i, Ci in enumerate(C_segs):
        Ci_prev = C_segs[(i - 1) % s]
        Ci_next = C_segs[(i + 1) % s]
        Ci_list = list(Ci)

        def po1(
            x: int,
            y: int,
            _p: set[int] = Ci_prev,
        ) -> bool:
            return (_nbrs(G, y) & _p).issubset(_nbrs(G, x) & _p)

        def po2(
            x: int,
            y: int,
            _n: set[int] = Ci_next,
        ) -> bool:
            return (_nbrs(G, x) & _n).issubset(_nbrs(G, y) & _n)

        for x, y in itertools.combinations(Ci_list, 2):
            if not (po1(x, y) or po1(y, x)):
                return False, (f"Vertices {x},{y} in C_{i} are incomparable under po1.")
            if not (po2(x, y) or po2(y, x)):
                return False, (f"Vertices {x},{y} in C_{i} are incomparable under po2.")
            if po1(x, y) and not po1(y, x) and not po2(x, y):
                return False, (f"Preorders po1, po2 disagree on ({x},{y}) in C_{i}.")
            if po1(y, x) and not po1(x, y) and not po2(y, x):
                return False, (f"Preorders po1, po2 disagree on ({y},{x}) in C_{i}.")

        order = _build_linear_order(Ci_list, po1, po2)
        if order is None:
            return False, (f"Cannot build a consistent linear order on C_{i}.")
        core_orders.append(order)

    # Step 5. Covered vertices and good segments.
    covered: set[int] = set()
    for i, Ci in enumerate(C_segs):
        Ci_next = C_segs[(i + 1) % s]
        order = core_orders[i]
        for idx_u in range(len(order)):
            u = order[idx_u]
            for idx_v in range(idx_u + 1, len(order)):
                v = order[idx_v]
                if (_nbrs(G, u) & Ci_next) == (_nbrs(G, v) & Ci_next):
                    covered.add(v)

    good_segs = [
        i
        for i, Ci in enumerate(C_segs)
        if all(any(not G.has_edge(u, w) for w in C_segs[(i + 1) % s]) for u in Ci)
    ]

    return True, ClockworkStructure(
        G=G,
        C_nodes=C_nodes,
        B_nodes=B_nodes,
        s=s,
        B_segments=B_segs,
        C_segments=C_segs,
        core_orders=core_orders,
        covered_vertices=covered,
        good_segments=good_segs,
    )


# -------------------------------------------------------------------
# Dominated-vertex removal
# -------------------------------------------------------------------


def remove_dominated_vertices(G: nx.Graph) -> nx.Graph:
    """Iteratively remove dominated vertices until none remain.

    By Theorem 2.3 of Larrion *et al.* (2004) this preserves
    clique-graph behaviour; by Theorem 3.3 the result is still a
    clockwork graph when *G* is one.

    .. rubric:: Parameters

    G : nx.Graph
        Input graph.

    .. rubric:: Returns

    nx.Graph
        A copy of *G* with all dominated vertices removed.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.clockwork import remove_dominated_vertices
    >>> H = remove_dominated_vertices(nx.cycle_graph(5))
    >>> sorted(H.nodes())
    [0, 1, 2, 3, 4]
    """
    H = G.copy()
    while True:
        u = find_dominated_vertex(H)
        if u is None:
            return H
        H.remove_node(u)


# -------------------------------------------------------------------
# Clique-divergence decision: Theorem 3.6
# -------------------------------------------------------------------


def is_clique_divergent_clockwork(
    G: nx.Graph,
) -> tuple[bool | None, str]:
    r"""Decide whether the clockwork graph *G* is clique-divergent.

    Uses Theorem 3.6 of Larrion *et al.* (2004):

    1. Remove dominated vertices (preserving clockwork structure).
    2. Recognise the residual graph.
    3. *G* is clique-divergent iff the residual has at least one
       good segment.

    .. rubric:: Parameters

    G : nx.Graph
        Input graph (should be clockwork for a definitive answer).

    .. rubric:: Returns

    tuple[bool | None, str]
        ``(True, explanation)`` when divergent,
        ``(False, explanation)`` when bounded, or
        ``(None, reason)`` when not recognised as clockwork.
    """
    H = remove_dominated_vertices(G)
    ok, result = recognize_clockwork(H)
    if not ok:
        assert isinstance(result, str)
        return None, (
            "After removing dominated vertices the "
            "residual graph is not recognised as a "
            f"clockwork graph: {result}"
        )
    assert isinstance(result, ClockworkStructure)
    g = len(result.good_segments)
    if g > 0:
        return True, (
            f"CLIQUE-DIVERGENT: residual has {g} good "
            f"segment(s) at {result.good_segments}."
        )
    return False, ("CLIQUE-BOUNDED: residual has 0 good segments.")


# -------------------------------------------------------------------
# Convenience wrapper
# -------------------------------------------------------------------


def recognize_as_clockwork_graph(
    G: nx.Graph,
) -> tuple[bool, ClockworkGraph | str]:
    """Recognise *G* and return a :class:`ClockworkGraph`.

    .. rubric:: Parameters

    G : nx.Graph
        Input graph.

    .. rubric:: Returns

    tuple[bool, ClockworkGraph | str]
        ``(True, clockwork_graph)`` on success;
        ``(False, reason)`` otherwise.

    .. rubric:: Examples

    >>> from pycliques.clockwork import (
    ...     clockwork_graph, recognize_as_clockwork_graph,
    ... )
    >>> G = clockwork_graph(
    ...     [1, 1, 1], [[1], [0], [0]], 2, [1, 0]
    ... )
    >>> ok, cg = recognize_as_clockwork_graph(G)
    >>> ok
    True
    """
    ok, result = recognize_clockwork(G)
    if not ok:
        assert isinstance(result, str)
        return False, result
    assert isinstance(result, ClockworkStructure)
    return True, ClockworkGraph.from_structure(result)
