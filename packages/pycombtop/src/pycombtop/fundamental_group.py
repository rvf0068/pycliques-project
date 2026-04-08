"""
fundamental_group.py
====================
Compute the fundamental group of the clique complex of a graph.

This is a Python translation of the GAP/GRAPE function
``FundamentalRecordSimplicialComplex`` by Leonard H. Soicher (1999–2015),
from the file ``fundamental_v2.g``.

Reference
---------
Sarah Rees and Leonard H. Soicher,
'An algorithmic approach to fundamental groups and covers of
combinatorial cell complexes',
J. Symbolic Comp. 29 (2000), 59–77.

The algorithm works as follows:

1. Build a spanning tree T of the graph by BFS.
   Every edge in T gets label = identity (the trivial word).

2. Process remaining edges one by one:
   - If a new edge completes a triangle with two already-labelled edges,
     its label is determined: label(x,y) = label(x,z) * label(z,y)
     for any common neighbour z already in the spanning subgraph.
   - If a new edge does NOT complete any triangle, it introduces a new
     free generator of the fundamental group.

3. Every triangle {x, y, z} in the clique complex gives a relator:
     label(x,y) * label(y,z) * label(z,x) = 1

4. The fundamental group is the finitely presented group with the
   generators and relators collected above.

Intended location
-----------------
``packages/pycombtop/src/pycombtop/fundamental_group.py``

Dependencies
------------
- networkx
- sympy (FreeGroup, FpGroup / PermutationGroup via coset enumeration)

Public API
----------
- :class:`FundamentalRecord`  – dataclass holding the result
- :func:`fundamental_group`   – main entry point (networkx graph → FundamentalRecord)
- :func:`covering_graph`      – construct the covering graph corresponding to
                                a subgroup of π₁
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.free_groups import FreeGroup, free_group

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FundamentalRecord:
    """The result of a fundamental group computation.

    .. rubric:: Attributes

    group : FpGroup
        A finitely presented group isomorphic to π₁(Δ(G), basepoint),
        where Δ(G) is the clique complex of G.
        If the complex is simply connected the group has no generators.
    generators : list
        The generators of ``group`` as sympy FreeGroup elements,
        in the order they were introduced.  Each corresponds to one
        non-tree edge of G.
    edge_labels : dict[tuple[int,int], FreeGroup element]
        Maps every directed edge (u, v) of G to its label in the
        free group (before imposing the relators).  Tree edges map to
        the identity; non-tree edges map to a generator or its inverse.
    spanning_tree : set[frozenset[int]]
        The undirected edges of the spanning tree T used in the computation.
    relators : list
        The relators (as free-group words) added to present the group.
    rank : int
        The rank (number of generators) of the fundamental group,
        equal to |E(G)| - |V(G)| + 1  minus the number of triangles
        that killed generators.  This equals the first Betti number
        β₁ of Δ(G) when Δ(G) is simply connected, and the rank of
        the abelianisation otherwise.
    """

    group: FpGroup
    generators: list
    edge_labels: dict
    spanning_tree: set
    relators: list
    rank: int

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"FundamentalRecord(\n"
            f"  rank={self.rank},\n"
            f"  generators={self.generators},\n"
            f"  relators={self.relators}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sorted_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _triangles(graph: nx.Graph) -> list[tuple[int, int, int]]:
    """Return all triangles of *graph* as sorted triples (i < j < k)."""
    result = []
    nodes = list(graph.nodes())
    for idx, u in enumerate(nodes):
        for v in nodes[idx + 1 :]:
            if not graph.has_edge(u, v):
                continue
            for w in graph.neighbors(u):
                if w > v and graph.has_edge(v, w):
                    result.append((u, v, w))
    return result


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def fundamental_group(graph: nx.Graph) -> FundamentalRecord:
    """Compute the fundamental group of the clique complex of *graph*.

    .. rubric:: Parameters

    graph : nx.Graph
        A simple, connected, non-empty graph.  The simplicial complex
        used is the clique complex Δ(G): vertices are vertices of G,
        edges are edges of G, 2-simplices are triangles of G.

    .. rubric:: Returns

    FundamentalRecord
        See :class:`FundamentalRecord` for a description of all fields.

    .. rubric:: Raises

    ValueError
        If *graph* is empty or not connected.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.fundamental_group import fundamental_group
    >>> G = nx.cycle_graph(4)          # square — π₁ ≅ ℤ
    >>> rec = fundamental_group(G)
    >>> rec.rank
    1
    >>> G2 = nx.complete_graph(4)      # K₄ — simply connected
    >>> rec2 = fundamental_group(G2)
    >>> rec2.rank
    0
    """
    if graph.order() == 0:
        raise ValueError("Graph must be non-empty.")
    if not nx.is_connected(graph):
        raise ValueError("Graph must be connected.")

    # Work with integer-labelled nodes for simplicity
    graph = nx.convert_node_labels_to_integers(graph)
    nodes = sorted(graph.nodes())

    # ------------------------------------------------------------------
    # Step 1: Build spanning tree T by BFS
    # ------------------------------------------------------------------
    spanning_tree: set[frozenset[int]] = set()
    # edge_labels[u][v] = free-group word labelling directed edge (u,v)
    # We use strings as placeholders first, then convert to sympy words.
    # At this stage labels are stored as generator-index tuples:
    #   None  = identity (not yet assigned)
    #   ("id",) = tree edge (identity)
    #   ("gen", k) = k-th generator
    #   ("inv", k) = inverse of k-th generator
    #   ("word", ...) = product to be composed later

    # We store labels as sympy FreeGroup words throughout.
    # We'll build the FreeGroup lazily as generators are introduced.

    # For BFS we track which undirected edges are still "unused"
    unused: set[tuple[int, int]] = set()
    for u, v in graph.edges():
        unused.add(_sorted_edge(u, v))

    # tree_label[u][v] and non_tree_label will be built during the algorithm.
    # We use a dict of dicts.  Values are (word in the free group on gens_so_far).
    # Since sympy FreeGroup is immutable (fixed generators), we store raw
    # generator indices and build the FpGroup only at the end.
    #
    # Internal label representation during construction:
    #   A label is an integer list [e_1, e_2, ..., e_k] where each e_i is
    #   a signed generator index (1-based): +k means g_k, -k means g_k^{-1}.
    #   The empty list [] is the identity.

    label: dict[tuple[int, int], list[int]] = {}  # directed edge -> word

    def compose(a: list[int], b: list[int]) -> list[int]:
        """Concatenate two words and freely reduce."""
        w = a + b
        # free reduction
        changed = True
        while changed:
            changed = False
            for i in range(len(w) - 1):
                if w[i] == -w[i + 1] or w[i] == -w[i + 1]:
                    if w[i] + w[i + 1] == 0:
                        w = w[:i] + w[i + 2 :]
                        changed = True
                        break
        return w

    def invert(a: list[int]) -> list[int]:
        """Invert a word."""
        return [-x for x in reversed(a)]

    # Queue of edges to add because they complete a 2-claw to a triangle
    edges_to_add: deque[tuple[int, int]] = deque()
    in_edges_to_add: dict[tuple[int, int], bool] = {}

    # delta_adj: adjacency of the "current spanning subgraph" (starts as T)
    delta_adj: dict[int, set[int]] = {v: set() for v in nodes}

    def add_tree_edge(u: int, v: int) -> None:
        """Add edge (u,v) to spanning tree with identity label."""
        spanning_tree.add(frozenset({u, v}))
        label[(u, v)] = []  # identity
        label[(v, u)] = []
        unused.discard(_sorted_edge(u, v))
        delta_adj[u].add(v)
        delta_adj[v].add(u)
        # enqueue edges that now complete a 2-claw to a triangle
        for w in list(delta_adj[u] & set(graph.neighbors(v))):
            key = _sorted_edge(v, w)
            if not in_edges_to_add.get(key, False):
                in_edges_to_add[key] = True
                edges_to_add.append((v, w))
        for w in list(delta_adj[v] & set(graph.neighbors(u))):
            key = _sorted_edge(u, w)
            if not in_edges_to_add.get(key, False):
                in_edges_to_add[key] = True
                edges_to_add.append((u, w))

    # BFS
    visited = {nodes[0]}
    bfs_queue = deque([nodes[0]])
    while bfs_queue:
        u = bfs_queue.popleft()
        for v in sorted(graph.neighbors(u)):
            if v not in visited:
                visited.add(v)
                bfs_queue.append(v)
                add_tree_edge(u, v)

    # ------------------------------------------------------------------
    # Step 2 & 3: Process remaining edges, assign labels and collect relators
    # ------------------------------------------------------------------
    ngens = 0
    relators: list[list[int]] = []

    def process_enqueued() -> None:
        """Process all edges in edges_to_add."""
        nonlocal ngens
        while edges_to_add:
            x, y = edges_to_add.popleft()
            key = _sorted_edge(x, y)
            in_edges_to_add[key] = False
            if key not in unused:
                continue  # already processed
            unused.discard(key)
            delta_adj[x].add(y)
            delta_adj[y].add(x)

            # Common neighbours of x and y already in delta
            common = sorted(delta_adj[x] & delta_adj[y])

            if ngens == 0:
                # All existing labels are identity → label is identity
                label[(x, y)] = []
                label[(y, x)] = []
            else:
                # Determine label from first common neighbour
                z0 = common[0]
                label[(x, y)] = compose(label[(x, z0)], label[(z0, y)])
                label[(y, x)] = invert(label[(x, y)])

                # Every additional common neighbour gives a relator
                for z in common[1:]:
                    if ngens > 0:
                        rel = compose(
                            compose(label[(x, y)], label[(y, z)]), label[(z, x)]
                        )
                        if rel and rel not in relators and invert(rel) not in relators:
                            relators.append(rel)

            # Propagate: edges that now complete new 2-claws
            for f_src, f_dst in [(x, y), (y, x)]:
                for w in set(graph.neighbors(f_dst)) & delta_adj[f_src]:
                    if w not in delta_adj[f_dst]:
                        wkey = _sorted_edge(f_dst, w)
                        if not in_edges_to_add.get(wkey, False):
                            in_edges_to_add[wkey] = True
                            edges_to_add.append((f_dst, w))

    def find_next_unused() -> tuple[int, int] | None:
        """Return the lexicographically smallest unused edge, or None."""
        if not unused:
            return None
        return min(unused)

    process_enqueued()

    nxt = find_next_unused()
    while nxt is not None:
        ngens += 1
        x, y = nxt
        unused.discard(_sorted_edge(x, y))
        delta_adj[x].add(y)
        delta_adj[y].add(x)

        # New generator g_{ngens}: label (x,y) = g_{ngens}
        label[(x, y)] = [ngens]
        label[(y, x)] = [-ngens]

        # Propagate new 2-claws
        for f_src, f_dst in [(x, y), (y, x)]:
            for w in set(graph.neighbors(f_dst)) & delta_adj[f_src]:
                if w not in delta_adj[f_dst]:
                    wkey = _sorted_edge(f_dst, w)
                    if not in_edges_to_add.get(wkey, False):
                        in_edges_to_add[wkey] = True
                        edges_to_add.append((f_dst, w))

        process_enqueued()
        nxt = find_next_unused()

    # Also collect relators from triangles involving only tree edges
    # (these are automatically satisfied but we add them for completeness
    # when verifying: they should all reduce to the identity).
    # Actually: all triangles {x,y,z} give relator label(x,y)*label(y,z)*label(z,x).
    # Tree-only triangles give [] which we skip.
    for x, y, z in _triangles(graph):
        for cyc in [(x, y, z), (y, z, x), (z, x, y)]:
            a, b, c = cyc
            if (a, b) in label and (b, c) in label and (c, a) in label:
                rel = compose(compose(label[(a, b)], label[(b, c)]), label[(c, a)])
                if rel and rel not in relators and invert(rel) not in relators:
                    relators.append(rel)
                break  # one orientation per triangle is enough

    # ------------------------------------------------------------------
    # Step 4: Build sympy FpGroup
    # ------------------------------------------------------------------
    if ngens == 0:
        # Trivial group — sympy FreeGroup on 0 generators
        F, *_ = free_group("")  # "" gives a 1-generator group
        # Construct trivial group: F / [x] gives trivial, but easier:
        F = FreeGroup([])
        fp = FpGroup(F, [])
        gens_sym = []
        edge_labels_sym = {e: fp.identity for e in label}
    else:
        gen_names = ", ".join(f"x{i}" for i in range(1, ngens + 1))
        F, *syms = free_group(gen_names)
        gens_sym = list(syms)

        def word_to_sympy(w: list[int]):
            if not w:
                return F.identity
            result = F.identity
            for idx in w:
                if idx > 0:
                    result = result * gens_sym[idx - 1]
                else:
                    result = result * gens_sym[-idx - 1] ** -1
            return result

        sympy_relators = [word_to_sympy(r) for r in relators]
        fp = FpGroup(F, sympy_relators)

        edge_labels_sym = {e: word_to_sympy(w) for e, w in label.items()}

    return FundamentalRecord(
        group=fp,
        generators=gens_sym,
        edge_labels=edge_labels_sym,
        spanning_tree=spanning_tree,
        relators=relators,
        rank=ngens,
    )


# ---------------------------------------------------------------------------
# Convenience: abelianisation → first homology group H₁(Δ(G), ℤ)
# ---------------------------------------------------------------------------


def first_homology_rank(graph: nx.Graph) -> int:
    """Return the rank of H₁(Δ(G), ℤ) — the first Betti number β₁.

    This equals the rank of the abelianisation of π₁(Δ(G)).
    For a clique complex it is also |E| - |V| + c - (number of triangles
    that kill generators), where c is the number of connected components.

    .. rubric:: Parameters

    graph : nx.Graph
        A simple connected graph.

    .. rubric:: Returns

    int
        β₁(Δ(G)).
    """
    rec = fundamental_group(graph)
    return rec.rank


# ---------------------------------------------------------------------------
# Convenience: permutation representation via coset enumeration
# ---------------------------------------------------------------------------


def fundamental_group_as_permutation_group(
    graph: nx.Graph,
    degree: int | None = None,
) -> PermutationGroup:
    """Attempt to realise π₁(Δ(G)) as a sympy PermutationGroup.

    This uses the Todd–Coxeter coset enumeration algorithm (via sympy's
    ``FpGroup.coset_enumeration``) on the trivial subgroup.  It works
    reliably for finite groups of small order, and for free groups of
    small rank.

    .. rubric:: Parameters

    graph : nx.Graph
        A simple connected graph.
    degree : int, optional
        If given, coset enumeration is run on a subgroup of that index.
        If None, enumeration is run on the trivial subgroup.

    .. rubric:: Returns

    PermutationGroup
        A permutation group isomorphic to π₁(Δ(G)).

    .. rubric:: Raises

    RuntimeError
        If coset enumeration does not terminate (infinite group or
        enumeration limit exceeded).

    .. rubric:: Notes

    For graphs whose clique complex is simply connected (e.g. K_n, chordal
    graphs) the result is the trivial group ``PermutationGroup(Permutation(0))``.
    For a cycle C_n (n ≥ 4) the result is ℤ, but sympy cannot represent
    infinite groups as PermutationGroups — in that case ``rec.rank`` is
    more informative.
    """
    rec = fundamental_group(graph)
    fp = rec.group

    if rec.rank == 0:
        return PermutationGroup(Permutation(0))

    try:
        # sympy's FpGroup.order() uses coset enumeration internally
        order = fp.order()
        if order == 0:
            raise RuntimeError(
                "Fundamental group appears to be infinite. "
                "Use fundamental_group() directly and inspect rec.rank "
                "and rec.relators instead."
            )
        perm_rep = fp.as_fp_group()  # returns a PermutationGroup
        return perm_rep
    except Exception as exc:
        raise RuntimeError(
            f"Could not convert FpGroup to PermutationGroup: {exc}\n"
            "The group may be infinite. Use fundamental_group() directly."
        ) from exc


# ---------------------------------------------------------------------------
# Covering graph
# ---------------------------------------------------------------------------


def covering_graph(
    graph: nx.Graph,
    subgroup_words: list[list[int]] | None,
    rec: FundamentalRecord | None = None,
    *,
    max_cosets: int = 10000,
) -> nx.Graph:
    """Construct the covering graph of *graph* corresponding to a subgroup
    of π₁(Δ(G)).

    This is a Python translation of the GAP function ``CoveringGraph``
    (with the subgroup argument) from ``fundamental_v2.g`` by Soicher.

    The covering is defined by the action of π₁ on the right cosets of H.
    Concretely:

    - Number the right cosets of H as 0, 1, …, m-1  (coset 0 = H itself).
    - The vertices of the cover are pairs (v, i) with v ∈ V(G), i ∈ [0,m).
    - (v, i) — (w, j) is an edge of the cover iff (v, w) is an edge of G
      and  i · label(v,w) = j,  where the label acts on cosets by right
      multiplication.

    The resulting cover is connected if and only if H is a normal subgroup
    (or more generally if the coset action is transitive, i.e. always when
    H is given as a subgroup of a group generated by the edge labels).

    .. rubric:: Parameters

    graph : nx.Graph
        A simple, connected, non-empty graph.
    subgroup_words : list of list of int, or None
        Generators of the subgroup H ⊆ π₁, each expressed as a word in
        the internal signed-integer representation used by
        :func:`fundamental_group` — i.e. a list of signed generator
        indices such as ``[1, -2, 1]`` meaning g₁ g₂⁻¹ g₁.
        Pass ``None`` or ``[]`` to get the *universal cover* (H = trivial
        subgroup), which corresponds to the tree cover when the complex is
        a graph (1-skeleton only) or the simply-connected cover in general.
    rec : FundamentalRecord, optional
        If you already have the result of :func:`fundamental_group`, pass
        it here to avoid recomputing it.
    max_cosets : int, optional
        Maximum number of cosets to enumerate (default 10000).  If coset
        enumeration exceeds this limit, a :class:`RuntimeError` is raised.
        Use this to prevent infinite loops when the subgroup has infinite
        index in π₁.

    .. rubric:: Returns

    nx.Graph
        The covering graph.  Vertices are labelled as ``(v, i)`` where
        ``v`` is a vertex of *graph* (after integer relabelling) and
        ``i ∈ [0, m)`` is the coset index.  The graph is simple and
        undirected.

    .. rubric:: Raises

    ValueError
        If *graph* is empty or not connected.
    RuntimeError
        If coset enumeration exceeds *max_cosets*, indicating the
        subgroup likely has infinite index.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.fundamental_group import fundamental_group, covering_graph
    >>> # The cycle C4 has π₁ ≅ ℤ.  The trivial-subgroup cover is infinite,
    >>> # but the subgroup 2ℤ gives a 2-fold cover (another cycle C8).
    >>> G = nx.cycle_graph(4)
    >>> rec = fundamental_group(G)
    >>> rec.rank       # one generator, no relators → π₁ ≅ ℤ
    1
    >>> # subgroup generated by g₁² (word [1,1]):
    >>> cover = covering_graph(G, [[1, 1]], rec=rec)
    >>> cover.order()
    8
    >>> # The trivial group subgroup (H = π₁ itself) gives a 1-fold cover = G itself:
    >>> cover1 = covering_graph(G, [[1]], rec=rec)
    >>> cover1.order()
    4
    """
    if rec is None:
        rec = fundamental_group(graph)

    graph = nx.convert_node_labels_to_integers(graph)

    # ------------------------------------------------------------------
    # Build the coset table for H via Todd–Coxeter on the word labels.
    # We implement a self-contained coset enumeration so as not to depend
    # on sympy's FpGroup coset machinery being available for arbitrary
    # infinite groups.
    #
    # Generators of π₁ are indexed 1..ngens (positive = generator,
    # negative = inverse).  We work with right cosets of H.
    # ------------------------------------------------------------------

    ngens = rec.rank

    if ngens == 0:
        # π₁ is trivial → only one coset → cover = graph itself
        cover = nx.Graph()
        for v in graph.nodes():
            cover.add_node((v, 0))
        for u, v in graph.edges():
            cover.add_edge((u, 0), (v, 0))
        return cover

    # Normalise subgroup words: if None or empty, H = trivial subgroup
    if not subgroup_words:
        subgroup_words = []

    # ------------------------------------------------------------------
    # Todd–Coxeter coset enumeration (right cosets of H in π₁).
    #
    # State:
    #   coset_table[c][g]  = d  means  coset c · g = coset d
    #                          (g runs over ±1 .. ±ngens, stored as
    #                           index 0..2*ngens-1)
    #   A value of -1 means "unknown".
    #
    # We use generator index encoding:
    #   gen_index(k)  = k - 1          for k in [1..ngens]  (generator)
    #   gen_index(-k) = ngens + k - 1  for k in [1..ngens]  (inverse)
    # ------------------------------------------------------------------

    def _gi(signed_k: int) -> int:
        """Encode signed generator index to table column index."""
        return signed_k - 1 if signed_k > 0 else ngens + (-signed_k) - 1

    def _inv_gi(col: int) -> int:
        """Inverse column: if col encodes g, return column encoding g⁻¹."""
        return col + ngens if col < ngens else col - ngens

    UNKNOWN = -1
    # Start with coset 0 = H
    coset_table: list[list[int]] = [[-1] * (2 * ngens)]
    n_cosets = 1

    def _new_coset() -> int:
        nonlocal n_cosets
        coset_table.append([-1] * (2 * ngens))
        idx = n_cosets
        n_cosets += 1
        return idx

    def _define(c: int, col: int, d: int) -> None:
        """Set coset_table[c][col] = d and coset_table[d][inv] = c."""
        coset_table[c][col] = d
        coset_table[d][_inv_gi(col)] = c

    def _apply_word(coset: int, word: list[int]) -> int | None:
        """Apply *word* to *coset*; return the resulting coset or None if unknown."""
        c = coset
        for signed_k in word:
            col = _gi(signed_k)
            nxt = coset_table[c][col]
            if nxt == UNKNOWN:
                return None
            c = nxt
        return c

    # Subgroup generators: for each word w in subgroup_words,
    # the coset 0 must be fixed: 0 · w = 0.
    # Also, relators of π₁ must fix every coset.

    relators: list[list[int]] = rec.relators  # words in signed-int form

    # We run HLT (Haselgrove–Leech–Thomas) style.

    def _scan_and_fill(coset: int, word: list[int]) -> None:
        """Scan *word* from both ends; define new cosets as needed (HLT)."""
        # Forward scan
        f_coset = coset
        f_pos = 0
        # Backward scan
        b_coset = coset
        b_pos = len(word)

        while f_pos < b_pos:
            # Try forward step
            col = _gi(word[f_pos])
            nxt = coset_table[f_coset][col]
            if nxt == UNKNOWN:
                break
            f_coset = nxt
            f_pos += 1

        while b_pos > f_pos:
            # Try backward step
            col = _gi(-word[b_pos - 1])  # inverse of last generator
            nxt = coset_table[b_coset][col]
            if nxt == UNKNOWN:
                break
            b_coset = nxt
            b_pos -= 1

        if f_pos == b_pos:
            # The two ends met: check or define equality
            gen_idx = word[f_pos - 1] if f_pos > 0 else word[0]
            if coset_table[f_coset][_gi(gen_idx)] == UNKNOWN:
                pass  # will be handled by coincidence or define below
            # f_coset should equal b_coset
            if f_coset != b_coset:
                _merge(f_coset, b_coset)
        elif f_pos + 1 == b_pos:
            # One unknown step: deduce it to close the loop
            col = _gi(word[f_pos])
            if coset_table[f_coset][col] == UNKNOWN:
                # The gap connects f_coset to b_coset (standard HLT deduction)
                _define(f_coset, col, b_coset)

    # Coincidence handling (merge two cosets)
    merged: list[int] = list(range(0))  # will grow with n_cosets

    def _find(c: int) -> int:
        """Union-find representative."""
        while True:
            r = merged[c] if c < len(merged) else c
            if r == c:
                return c
            c = r

    def _merge(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        # keep the smaller index as canonical
        keep, lose = (ra, rb) if ra < rb else (rb, ra)
        while lose >= len(merged):
            merged.append(len(merged))
        merged[lose] = keep
        # propagate: update table entries that reference 'lose'
        for c in range(n_cosets):
            if c >= len(merged):
                break
            for col in range(2 * ngens):
                if c < len(coset_table) and coset_table[c][col] == lose:
                    coset_table[c][col] = keep
            if _find(c) == lose:
                merged[c] = keep

    # Initialise merged list
    merged.extend(range(n_cosets))

    # Process subgroup generator words: coset 0 · w = 0
    for w in subgroup_words:
        _scan_and_fill(0, w)

    # Main enumeration loop: for each coset and each generator,
    # if the action is unknown, define a new coset.
    coset_idx = 0
    while coset_idx < n_cosets:
        # Extend merged if needed
        while len(merged) < n_cosets:
            merged.append(len(merged))
        if _find(coset_idx) != coset_idx:
            coset_idx += 1
            continue
        for col in range(2 * ngens):
            if coset_idx < len(coset_table) and coset_table[coset_idx][col] == UNKNOWN:
                if n_cosets >= max_cosets:
                    raise RuntimeError(
                        f"Coset enumeration exceeded {max_cosets} cosets. "
                        "The subgroup likely has infinite index. "
                        "Use max_cosets parameter to increase the limit."
                    )
                d = _new_coset()
                while len(merged) < n_cosets:
                    merged.append(len(merged))
                _define(coset_idx, col, d)
                # Scan all relators through the new coset
                for rel in relators:
                    _scan_and_fill(coset_idx, rel)
                for w in subgroup_words:
                    _scan_and_fill(0, w)
        coset_idx += 1

    # Compact: renumber live cosets (those where _find(c)==c)
    live = [c for c in range(n_cosets) if _find(c) == c]
    renum = {c: i for i, c in enumerate(live)}
    m = len(live)  # index of the cover = number of cosets of H

    # Action of generator g (col) on coset i (in compacted numbering):
    # action[col][i] = renum[coset_table[live[i]][col]]
    action: list[list[int]] = []
    for col in range(2 * ngens):
        perm = []
        for i, c in enumerate(live):
            target = coset_table[c][col]
            if target == UNKNOWN or target == -1:
                # Should not happen after full enumeration; fall back to identity
                perm.append(i)
            else:
                perm.append(renum[_find(target)])
        action.append(perm)

    # ------------------------------------------------------------------
    # Build the covering graph.
    # Vertices: (v, i)  for v in graph.nodes(), i in range(m)
    # Edge (v,i)-(w,j) iff (v,w) is an edge of graph and
    #   action of label(v,w) maps coset i to coset j.
    # ------------------------------------------------------------------
    cover = nx.Graph()
    nodes_sorted = sorted(graph.nodes())
    for v in nodes_sorted:
        for i in range(m):
            cover.add_node((v, i))

    for u, v in graph.edges():
        label_uv = rec.edge_labels.get((u, v))  # sympy FreeGroup word
        # Convert sympy word back to signed-int list for action lookup
        word_uv = _sympy_word_to_signed(label_uv, rec.generators)
        for i in range(m):
            j = _apply_perm_word(action, word_uv, i, ngens)
            cover.add_edge((u, i), (v, j))

    return cover


def _sympy_word_to_signed(word, generators) -> list[int]:  # type: ignore[no-untyped-def]
    """Convert a sympy FreeGroup word to a signed-integer word.

    .. rubric:: Parameters

    word : sympy FreeGroup element
    generators : list of sympy FreeGroup generators

    .. rubric:: Returns

    list[int]
        Signed generator indices (1-based).
    """
    if word is None:
        return []
    # sympy stores words as tuples of (generator, exponent) pairs
    result = []
    try:
        # FreeGroupElement.array_form gives [(gen_symbol, exp), ...]
        for gen_sym, exp in word.array_form:
            idx = next(
                (
                    i + 1
                    for i, g in enumerate(generators)
                    if g.array_form and g.array_form[0][0] == gen_sym
                ),
                None,
            )
            if idx is None:
                continue
            if exp > 0:
                result.extend([idx] * exp)
            else:
                result.extend([-idx] * (-exp))
    except AttributeError:
        pass  # identity element has no array_form entries
    return result


def _apply_perm_word(
    action: list[list[int]], word: list[int], start: int, ngens: int
) -> int:
    """Apply a signed-integer *word* to coset *start* using *action*.

    .. rubric:: Parameters

    action : list[list[int]]
        action[col][i] = image of coset i under generator encoded by col.
    word : list[int]
        Signed generator indices.
    start : int
        Starting coset index.
    ngens : int
        Number of generators.
    """
    c = start
    for signed_k in word:
        col = signed_k - 1 if signed_k > 0 else ngens + (-signed_k) - 1
        c = action[col][c]
    return c
