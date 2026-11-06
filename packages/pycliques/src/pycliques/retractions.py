"""Graph retractions and induced subgraphs."""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism

from .cliques import clique_graph
from .coaffinations import automorphisms
from .dominated import closed_neighborhood, completely_pared_graph
from .named import complement_of_cycle, octahedron, suspension_of_cycle

# ---------------------------------------------------------------------------
# Logging Setup: The "NullHandler" prevents logs from polluting doctests
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def dict_to_tuple(
    the_dict: Mapping[Hashable, Any],
) -> tuple[tuple[Hashable, Any], ...]:
    """Convert a mapping to a tuple of key-value pairs in iteration order.

    .. rubric:: Parameters

    the_dict : collections.abc.Mapping
        Mapping to convert. Its iteration order is preserved.

    .. rubric:: Returns

    tuple[tuple[Hashable, Any], ...]
        Tuple of ``(key, value)`` pairs.

    .. rubric:: Examples

    >>> from pycliques.retractions import dict_to_tuple
    >>> dict_to_tuple({2: 'b', 1: 'a'})
    ((2, 'b'), (1, 'a'))
    """
    return tuple(the_dict.items())


def invert_dict(
    the_dict: Mapping[Hashable, Hashable],
) -> dict[Hashable, Hashable]:
    """Invert a mapping whose values are unique.

    The input must be injective: no two keys may have the same value.
    Under that assumption, the returned mapping is the inverse on the image
    of the input.

    .. rubric:: Parameters

    the_dict : collections.abc.Mapping
        Mapping whose keys and values will be swapped.

    .. rubric:: Returns

    dict[Hashable, Hashable]
        A new dictionary with keys and values exchanged.

    .. rubric:: Examples

    >>> from pycliques.retractions import invert_dict
    >>> invert_dict({0: 'a', 1: 'b'})
    {'a': 0, 'b': 1}
    """
    # Dictionary comprehension is the fastest way to build a dict in Python
    return {v: k for k, v in the_dict.items()}


def graph_from_gap_adjacency_list(the_list: list[list[int]]) -> nx.Graph:
    """Create a 0-indexed graph from a 1-indexed GAP adjacency list.

    .. rubric:: Parameters

    the_list : list[list[int]]
        Adjacency list where vertex indices start at 1 (GAP convention).

    .. rubric:: Returns

    networkx.Graph
        Graph with 0-indexed vertices.

    .. rubric:: Examples

    >>> from pycliques.retractions import graph_from_gap_adjacency_list
    >>> g = graph_from_gap_adjacency_list([[2], [1]])
    >>> sorted(g.edges())
    [(0, 1)]
    """
    graph = nx.Graph()
    for i, adj in enumerate(the_list):
        graph.add_edges_from([(i, v - 1) for v in adj])
    return graph


def is_map(
    domain: nx.Graph,
    codomain: nx.Graph,
    ismap: Mapping[Hashable, Hashable],
) -> bool:
    """Determine whether ``ismap`` defines a graph homomorphism.

    It is not required that every vertex in ``domain`` has a value
    determined in ``ismap``.

    .. rubric:: Parameters

    domain : networkx.Graph
        Source graph.
    codomain : networkx.Graph
        Target graph.
    ismap : dict
        Partial or total vertex mapping from ``domain`` to ``codomain``.

    .. rubric:: Returns

    bool
        ``True`` if the mapping preserves adjacency.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import is_map
    >>> mapping = {0: 0, 1: 1, 2: 0, 3: 1}
    >>> is_map(nx.cycle_graph(4), nx.complete_graph(2), mapping)
    True
    >>> mapping = {0: 0, 1: 1}
    >>> is_map(nx.cycle_graph(4), nx.complete_graph(2), mapping)
    True
    """
    for u, v in domain.edges():
        if u in ismap and v in ismap:
            wu, wv = ismap[u], ismap[v]
            # Must map to an edge, or collapse to a single vertex
            if wu != wv and not codomain.has_edge(wu, wv):
                return False
    return True


def _extension_of_map(
    large: nx.Graph,
    small: nx.Graph,
    mapp: Mapping[Hashable, Hashable],
    v: Hashable,
) -> set[Hashable]:
    """Find the set of vertices of `small` that could be images of `v`."""
    common = set(small.nodes())
    for w in set(large[v]).intersection(mapp.keys()):
        common &= closed_neighborhood(small, mapp[w])
    return common


def _extend_retraction(
    large: nx.Graph,
    small: nx.Graph,
    state: tuple[tuple[Hashable, Hashable], ...],
) -> Iterator[tuple[tuple[Hashable, Hashable], ...]]:
    """Generator to backtrack and complete a retraction from large to small."""
    ret = dict(state)
    remaining = list(set(large.nodes()) - set(ret.keys()))

    for v in remaining:
        for w in _extension_of_map(large, small, ret, v):
            if len(ret) == large.order() - 1:
                yield ((v, w),)
            else:
                for res in _extend_retraction(large, small, state + ((v, w),)):
                    yield ((v, w),) + res


class RetractionSearchStatus(Enum):
    """Completion state for bounded retraction search."""

    FOUND = auto()
    EXHAUSTED = auto()
    LIMIT_REACHED = auto()


@dataclass(frozen=True)
class BoundedRetractionResult:
    """A bounded retraction result with explicit search-completion status."""

    status: RetractionSearchStatus
    retraction: dict[Hashable, Hashable] | None
    inclusion: dict[Hashable, Hashable] | None
    states_explored: int
    assignments_tried: int


@dataclass
class _RetractionSearchFrame:
    vertex: Hashable
    candidates: list[Hashable]
    next_candidate: int = 0
    assigned: bool = False


def bounded_retraction(
    large: nx.Graph,
    small: nx.Graph,
    *,
    max_states: int = 100_000,
    max_assignments: int = 1_000_000,
) -> BoundedRetractionResult:
    """Search for a retraction using bounded backtracking, without automorphisms.

    ``small`` must be an induced subgraph of ``large`` with the same vertex
    labels. The target vertices are fixed pointwise; other vertices are
    assigned to target vertices so each source edge maps to an edge or is
    collapsed to one vertex.

    The operation limits count search states and tested
    vertex-to-target assignments. Reaching either limit returns
    ``LIMIT_REACHED``; only a fully searched assignment tree returns
    ``EXHAUSTED``.
    """
    if max_states < 1:
        raise ValueError("max_states must be at least 1")
    if max_assignments < 1:
        raise ValueError("max_assignments must be at least 1")
    target_nodes = set(small)
    if not target_nodes <= set(large):
        raise ValueError("small must be an induced subgraph of large")
    induced = large.subgraph(target_nodes)
    if {frozenset(edge) for edge in induced.edges()} != {
        frozenset(edge) for edge in small.edges()
    }:
        raise ValueError("small must be an induced subgraph of large")

    inclusion = {vertex: vertex for vertex in small}
    assignment: dict[Hashable, Hashable] = dict(inclusion)
    target_order = list(small)
    states_explored = 0
    assignments_tried = 0
    limit_reached = False

    def candidates_for(vertex: Hashable) -> list[Hashable]:
        candidates = set(target_order)
        for neighbor in large.neighbors(vertex):
            if neighbor in assignment:
                candidates &= closed_neighborhood(small, assignment[neighbor])
        return [target for target in target_order if target in candidates]

    def enter_state() -> tuple[bool, bool]:
        """Return (complete, valid) after recording a newly entered state."""
        nonlocal states_explored
        states_explored += 1
        remaining = [vertex for vertex in large if vertex not in assignment]
        if not remaining:
            valid = (
                set(assignment) == set(large)
                and set(assignment.values()) <= target_nodes
                and is_map(large, small, assignment)
                and all(assignment[vertex] == vertex for vertex in small)
            )
            return True, valid
        vertex_options = [(vertex, candidates_for(vertex)) for vertex in remaining]
        vertex, candidates = min(vertex_options, key=lambda item: len(item[1]))
        frames.append(_RetractionSearchFrame(vertex, candidates))
        return False, False

    frames: list[_RetractionSearchFrame] = []
    complete, valid = enter_state()
    if complete and valid:
        return BoundedRetractionResult(
            RetractionSearchStatus.FOUND,
            dict(assignment),
            inclusion,
            states_explored,
            assignments_tried,
        )

    while frames:
        frame = frames[-1]
        if frame.assigned:
            del assignment[frame.vertex]
            frame.assigned = False
        if frame.next_candidate >= len(frame.candidates):
            frames.pop()
            continue
        if assignments_tried >= max_assignments:
            limit_reached = True
            break

        target = frame.candidates[frame.next_candidate]
        frame.next_candidate += 1
        assignments_tried += 1
        assignment[frame.vertex] = target
        frame.assigned = True

        if states_explored >= max_states:
            limit_reached = True
            break
        complete, valid = enter_state()
        if complete and valid:
            return BoundedRetractionResult(
                RetractionSearchStatus.FOUND,
                dict(assignment),
                inclusion,
                states_explored,
                assignments_tried,
            )

    return BoundedRetractionResult(
        RetractionSearchStatus.LIMIT_REACHED
        if limit_reached
        else RetractionSearchStatus.EXHAUSTED,
        None,
        None,
        states_explored,
        assignments_tried,
    )


def retraction(large: nx.Graph, small: nx.Graph) -> Iterator[tuple[dict, dict]]:
    """Yield all retractions from ``large`` onto ``small``.

    Each retraction is a pair of dictionaries:

    1. A surjective map from ``large`` to ``small`` (the retraction).
    2. A map from ``small`` to ``large`` (the inclusion).

    .. rubric:: Parameters

    large : networkx.Graph
        Graph that may retract.
    small : networkx.Graph
        Target subgraph.

    .. rubric:: Yields

    tuple[dict, dict]
        ``(retraction_map, inclusion_map)`` pairs.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import retraction
    >>> list(retraction(nx.wheel_graph(4), nx.cycle_graph(4)))
    []
    >>> list(retraction(nx.path_graph(3), nx.path_graph(2)))
    [({0: 0, 1: 1, 2: 0}, {0: 0, 1: 1}), ({0: 0, 1: 1, 2: 1}, {0: 0, 1: 1})]
    """
    GM = isomorphism.GraphMatcher(large, small)
    rets = GM.subgraph_isomorphisms_iter()

    a_small = list(automorphisms(small))
    a_large = list(automorphisms(large))

    # PERFORMANCE FIX: Use a set of frozensets for O(1) mathematical symmetry checks
    seen_signatures = set()

    for ret in rets:
        ret_sig = frozenset(ret.items())

        if ret_sig not in seen_signatures:
            if large.order() == small.order():
                yield (ret, invert_dict(ret))
            else:
                state = tuple(ret.items())
                for ext in _extend_retraction(large, small, state):
                    yield (dict(state + ext), invert_dict(ret))

            # Add all symmetric equivalent inclusions to the seen set
            for auto_s in a_small:
                for auto_l in a_large:
                    sym_sig = frozenset((auto_l[x], auto_s[ret[x]]) for x in ret)
                    seen_signatures.add(sym_sig)


def retracts(large: nx.Graph, small: nx.Graph) -> tuple[dict, dict] | bool:
    """Find one retraction without enumerating host automorphisms.

    The full `retraction` generator retains its historical enumeration
    semantics. For a first witness, its eager automorphism lists were
    wasteful and could be factorial in the size of `large`.

    Each induced embedding of `small` into `large` fixes the retraction
    on its image. A complete MRV backtracking search assigns the remaining
    source vertices to target vertices. Adjacency may map to adjacency
    *or equality*, as required by the graph-morphism convention.

    Returns `(retraction_map, inclusion_map)`, or `False`.
    """
    if large.number_of_nodes() < small.number_of_nodes():
        return False
    if large.number_of_edges() < small.number_of_edges():
        return False

    target_nodes = tuple(small)
    allowed_neighbors = {v: set(small[v]) | {v} for v in small}
    host_nodes = set(large)

    matcher = isomorphism.GraphMatcher(large, small)
    for embedding in matcher.subgraph_isomorphisms_iter():
        # GraphMatcher's dictionary maps host vertices into target labels,
        # and is induced on the embedded vertex set.
        assignment = dict(embedding)
        inclusion = invert_dict(embedding)
        remaining = host_nodes - set(assignment)

        def finish(unassigned: set) -> dict | None:
            if not unassigned:
                return dict(assignment)

            best_vertex = None
            best_options = None
            for vertex in unassigned:
                options = set(target_nodes)
                for neighbor in large[vertex]:
                    if neighbor in assignment:
                        options.intersection_update(
                            allowed_neighbors[assignment[neighbor]]
                        )
                        if not options:
                            return None
                if best_options is None or len(options) < len(best_options):
                    best_vertex, best_options = vertex, options
                    if len(options) == 1:
                        break

            assert best_vertex is not None and best_options is not None
            for image in best_options:
                assignment[best_vertex] = image
                result = finish(unassigned - {best_vertex})
                if result is not None:
                    return result
                del assignment[best_vertex]
            return None

        witness = finish(remaining)
        if witness is not None:
            return witness, inclusion
    return False


def verify_retraction_witness(
    large: nx.Graph,
    small: nx.Graph,
    witness: tuple[dict, dict],
) -> bool:
    """Verify an induced inclusion and graph-morphism retraction.

    This checker does not invoke a search or enumerate automorphisms.
    Adjacency may map to adjacency or equality, as in clique graph theory.
    """
    if not isinstance(witness, tuple) or len(witness) != 2:
        return False
    projection, inclusion = witness
    if not isinstance(projection, dict) or not isinstance(inclusion, dict):
        return False
    if set(projection) != set(large) or set(inclusion) != set(small):
        return False
    if set(projection.values()) - set(small):
        return False
    if len(set(inclusion.values())) != len(small):
        return False
    if any(inclusion[v] not in large or projection[inclusion[v]] != v for v in small):
        return False
    if any(
        (large.has_edge(inclusion[u], inclusion[v]) != small.has_edge(u, v))
        for u, v in itertools.combinations(small, 2)
    ):
        return False
    return all(
        projection[u] == projection[v] or small.has_edge(projection[u], projection[v])
        for u, v in large.edges()
    )


def retracts_to(subgraph: nx.Graph) -> Callable[[nx.Graph], tuple[dict, dict] | bool]:
    """Return a function that checks retraction to ``subgraph``.

    .. rubric:: Parameters

    subgraph : networkx.Graph
        Fixed target subgraph.

    .. rubric:: Returns

    Callable
        A function ``f(g)`` returning the retraction or ``False``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import retracts_to
    >>> checker = retracts_to(nx.path_graph(2))
    >>> checker(nx.path_graph(3)) is not False
    True
    """
    return lambda g: retracts(g, subgraph)


def has_induced(large: nx.Graph, small: nx.Graph) -> dict | bool:
    """Return an injective map witnessing ``small`` as an induced subgraph.

    .. rubric:: Parameters

    large : networkx.Graph
        Host graph.
    small : networkx.Graph
        Pattern graph.

    .. rubric:: Returns

    dict | bool
        A node mapping from ``large`` to ``small`` if one exists,
        otherwise ``False``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import has_induced
    >>> has_induced(nx.complete_graph(4), nx.path_graph(3))
    False
    >>> has_induced(nx.cycle_graph(4), nx.complete_graph(3)) is False
    True
    """
    GM = isomorphism.GraphMatcher(large, small)
    try:
        result: dict = next(GM.subgraph_isomorphisms_iter())
        return result
    except StopIteration:
        return False


def _is_maximal_clique(graph: nx.Graph, clique: Iterable) -> bool:
    """Return True if `clique` is maximal (cannot be extended)."""
    clique_set = set(clique)

    # Candidates for extending the clique are all vertices currently outside it
    candidates = set(graph.nodes) - clique_set

    # Intersect the candidates with the neighborhood of each vertex in the clique
    for v in clique_set:
        candidates &= set(graph[v])
        if not candidates:
            # If no outside vertex is connected to all elements so far,
            # the clique is definitely maximal.
            return True

    # If candidates is not empty, those vertices are connected to EVERYTHING
    # in the clique, meaning the clique can be extended and is NOT maximal.
    return False


@dataclass(frozen=True)
class SpecialOctahedronWitness:
    """An induced special octahedron, with aligned opposite vertex pairs.

    The entries in `clique` form a maximal clique of the host graph.
    Each `antipodes[i]` is its unique nonneighbor among these entries;
    the antipodes themselves form a clique. This directly certifies O_d,
    where d = len(clique) >= 3.
    """

    clique: tuple[Hashable, ...]
    antipodes: tuple[Hashable, ...]

    @property
    def dimension(self) -> int:
        return len(self.clique)


def verify_special_octahedron_witness(
    graph: nx.Graph, witness: SpecialOctahedronWitness
) -> bool:
    """Verify a positive special-octahedron witness without doing any search."""
    if not isinstance(witness, SpecialOctahedronWitness):
        return False
    clique, antipodes = witness.clique, witness.antipodes
    d = len(clique)
    if d < 3 or len(antipodes) != d:
        return False
    vertices = clique + antipodes
    if len(set(vertices)) != 2 * d or not set(vertices) <= set(graph):
        return False
    if any(not graph.has_edge(u, v) for u, v in itertools.combinations(clique, 2)):
        return False
    if any(all(graph.has_edge(x, c) for c in clique) for x in graph if x not in clique):
        return False  # The selected clique must be maximal in the host.
    if any(
        graph.has_edge(clique[i], antipodes[i])
        or any(not graph.has_edge(antipodes[i], clique[j]) for j in range(d) if j != i)
        for i in range(d)
    ):
        return False
    return all(graph.has_edge(a, b) for a, b in itertools.combinations(antipodes, 2))


@dataclass
class _SpecialOctahedronFrame:
    """One pending branch of iterative Bron--Kerbosch exploration."""

    clique: tuple[int, ...]
    clique_mask: int
    candidates: int
    excluded: int
    options: int | None = None


def find_special_octahedron_witness(
    graph: nx.Graph,
) -> SpecialOctahedronWitness | None:
    """Find a special octahedron with exact, nonrecursive clique exploration.

    A special O_d has a maximal host clique C with one antipode for each
    v in C. Every partial clique P of C must already admit candidates for
    these antipodes, and those domains only shrink as P grows. Thus an
    empty domain safely prunes the entire search branch.

    The Bron--Kerbosch traversal and antipode assignment both use
    explicit stacks to avoid recursion-limit failures for large graphs.
    Adjacency masks exclude self-loops, as the clique relation does.
    None means exhaustive absence, never a resource timeout.
    """
    if graph.number_of_nodes() < 6:
        return None
    vertices = tuple(graph)
    count = len(vertices)
    indices = {v: i for i, v in enumerate(vertices)}
    neighbors = tuple(
        sum(1 << indices[w] for w in graph[v] if indices[w] != i)
        for i, v in enumerate(vertices)
    )
    full_mask = (1 << count) - 1

    def possible_antipodes(
        clique: tuple[int, ...], clique_mask: int
    ) -> list[int] | None:
        """Compute the necessary antipode domains of a partial clique."""
        prefix = [full_mask]
        for vertex in clique:
            prefix.append(prefix[-1] & neighbors[vertex])
        suffix = [full_mask] * (len(clique) + 1)
        for i in range(len(clique) - 1, -1, -1):
            suffix[i] = suffix[i + 1] & neighbors[clique[i]]
        domains = []
        for i, vertex in enumerate(clique):
            candidates = (
                prefix[i]
                & suffix[i + 1]
                & ~neighbors[vertex]
                & ~clique_mask
                & full_mask
            )
            if not candidates:
                return None
            domains.append(candidates)
        return domains

    def compatible_antipodes(
        clique: tuple[int, ...], domains: list[int]
    ) -> SpecialOctahedronWitness | None:
        """Find a pairwise-adjacent transversal of antipode domains."""
        order = sorted(range(len(clique)), key=lambda j: domains[j].bit_count())
        selected: list[int] = []
        common = [full_mask]
        options = [domains[order[0]]]
        while True:
            position = len(selected)
            if position == len(order):
                return SpecialOctahedronWitness(
                    clique=tuple(vertices[clique[j]] for j in order),
                    antipodes=tuple(vertices[j] for j in selected),
                )
            remaining = options[position]
            if not remaining:
                if not position:
                    return None
                options.pop()
                selected.pop()
                common.pop()
                continue

            bit = remaining & -remaining
            options[position] ^= bit
            candidate = bit.bit_length() - 1
            next_common = common[position] & neighbors[candidate]
            if not all(domains[j] & next_common for j in order[position + 1 :]):
                continue
            selected.append(candidate)
            common.append(next_common)
            if position + 1 < len(order):
                options.append(domains[order[position + 1]] & next_common)

    stack = [_SpecialOctahedronFrame((), 0, full_mask, 0)]
    while stack:
        frame = stack[-1]
        if frame.options is None:
            if len(frame.clique) + frame.candidates.bit_count() < 3:
                stack.pop()
                continue
            domains = (
                possible_antipodes(frame.clique, frame.clique_mask)
                if frame.clique
                else []
            )
            if domains is None:
                stack.pop()
                continue
            if not frame.candidates and not frame.excluded:
                if len(frame.clique) >= 3:
                    witness = compatible_antipodes(frame.clique, domains)
                    if witness is not None:
                        return witness
                stack.pop()
                continue

            pivot_set = frame.candidates | frame.excluded
            if pivot_set:
                pivot_nodes = (i for i in range(count) if pivot_set & (1 << i))
                pivot = max(
                    pivot_nodes,
                    key=lambda i: (frame.candidates & neighbors[i]).bit_count(),
                )
                frame.options = frame.candidates & ~neighbors[pivot]
            else:
                frame.options = frame.candidates
            continue

        if not frame.options:
            stack.pop()
            continue
        bit = frame.options & -frame.options
        frame.options ^= bit
        vertex = bit.bit_length() - 1
        child_candidates = frame.candidates & neighbors[vertex]
        child_excluded = frame.excluded & neighbors[vertex]
        frame.candidates ^= bit
        frame.excluded |= bit
        stack.append(
            _SpecialOctahedronFrame(
                frame.clique + (vertex,),
                frame.clique_mask | bit,
                child_candidates,
                child_excluded,
            )
        )
    return None


def _find_special_octahedra(
    graph: nx.Graph,
) -> tuple[bool, int | None]:
    """Return existence and dimension of a special octahedron."""
    witness = find_special_octahedron_witness(graph)
    return (True, witness.dimension) if witness is not None else (False, None)


def is_O3_free(graph: nx.Graph) -> bool:
    """Return whether *graph* contains no induced octahedron :math:`O_3`.

    This is the plain induced-subgraph check.  It is intentionally a thin
    wrapper around :func:`has_induced` and ``octahedron(3)``; it is the
    correct API for the question "Does *graph* contain an induced octahedron?"

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Returns

    bool
        ``True`` if the graph has no induced :math:`O_3`.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.named import octahedron
    >>> from pycliques.retractions import is_O3_free
    >>> is_O3_free(nx.octahedral_graph())
    False
    >>> is_O3_free(nx.complete_graph(4))
    True
    """
    return not has_induced(graph, octahedron(3))


def special_octahedra(graph: nx.Graph) -> bool:
    """Check for the stricter *special* octahedron condition.

    Warning:
        This is not the same as testing whether ``graph`` has an induced
        octahedron :math:`O_3`.  It is a stricter divergence/retraction-oriented
        property tailored to the special-octahedra arguments in this project:
        it looks for an induced :math:`O_n` with at least one triangle also
        maximal in the whole host graph.  Use :func:`is_O3_free` for the plain
        "no induced :math:`O_3`" test.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Returns

    bool
        ``True`` if ``graph`` contains a special octahedron.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import special_octahedra
    >>> special_octahedra(nx.octahedral_graph())
    True
    >>> special_octahedra(nx.cycle_graph(5))
    False
    """
    found, _ = _find_special_octahedra(graph)
    return found


def special_octahedra_dimension(graph: nx.Graph) -> int | None:
    """Return the dimension of a special octahedron in *graph*, if any.

    The dimension equals the number of pairs of antipodal vertices in the
    octahedron (e.g. the standard octahedron has dimension 3).

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Returns

    int or None
        Dimension of the first special octahedron found, or ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.retractions import special_octahedra_dimension
    >>> special_octahedra_dimension(nx.octahedral_graph())
    3
    >>> special_octahedra_dimension(nx.cycle_graph(5)) is None
    True
    """
    _, dim = _find_special_octahedra(graph)
    return dim


# ---------------------------------------------------------------------------
# CLI Application Logic
# ---------------------------------------------------------------------------


def _setup_logging(loglevel: int | None):
    """Safely setup logging ONLY when running as a CLI script."""
    if loglevel is not None:
        logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
        logging.basicConfig(
            level=loglevel,
            stream=sys.stdout,
            format=logformat,
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _string_to_graph(string: str) -> nx.Graph:
    """Parse a short graph descriptor into a NetworkX graph.

    Recognized prefixes: ``sc`` (suspension of cycle), ``cc`` (complement
    of cycle), ``o`` (octahedron).
    """
    if string.startswith("sc"):
        return suspension_of_cycle(int(string[2:]))
    elif string.startswith("cc"):
        return complement_of_cycle(int(string[2:]))
    elif string.startswith("o"):
        return octahedron(int(string[1:]))
    raise ValueError(f"Unknown graph string format: {string}")


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the find-retractions script."""
    from pycliques import __version__

    parser = argparse.ArgumentParser(description="Find retractions to subgraphs.")
    parser.add_argument(
        "--version", action="version", version=f"pycliques {__version__}"
    )

    parser.add_argument(
        "n", help="Number of clique graph iterations", type=int, metavar="INT"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="Set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )

    # The flag to trigger the fast algorithm
    parser.add_argument(
        "--special",
        action="store_true",
        help="Use the fast algorithm for special octahedra",
    )

    parser.add_argument(
        "large", help="Large graph in g6 format", type=str, metavar="STR"
    )

    # Make 'small' optional natively, so --special doesn't require a dummy string
    parser.add_argument(
        "small",
        nargs="?",
        default=None,
        help="Small graph string (e.g. o3, sc4). Required unless --special is used.",
        type=str,
        metavar="STR",
    )

    parsed = parser.parse_args(args)

    # Manually enforce the requirement only if --special is NOT used
    if not parsed.special and parsed.small is None:
        parser.error(
            "The 'small' graph argument is required for general "
            "retractions (e.g., 'o3')."
        )

    return parsed


def _main(args: list[str]):
    """Run the retraction search from parsed CLI arguments."""
    args_parsed = _parse_args(args)
    _setup_logging(args_parsed.loglevel)

    _logger.info("Parsing input graphs...")
    large = nx.from_graph6_bytes(bytes(args_parsed.large, "utf8"))

    # Iterate the operator
    for i in range(args_parsed.n):
        _logger.info(f"Iterating the clique operator (Step {i + 1})")
        large = completely_pared_graph(clique_graph(large))

    large = nx.convert_node_labels_to_integers(large)
    _logger.info(f"The large graph has order {large.order()}")
    _logger.info("Searching for retractions...")

    # --- The Unified Routing Logic ---
    if args_parsed.special:
        _logger.info("Using fast combinatorial check for special octahedra...")
        if special_octahedra(large):
            print("Found special octahedra retraction!")
        else:
            print("Sorry, could not find it!")
    else:
        _logger.info("Using general graph matcher...")
        small = _string_to_graph(args_parsed.small)
        has_retraction = retracts(large, small)
        if has_retraction:
            print(f"Found {has_retraction}")
        else:
            print("Sorry, could not find it!")

    _logger.info("Script ends here")


def main():  # pragma: no cover
    """Entry point for console_scripts."""
    _main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
