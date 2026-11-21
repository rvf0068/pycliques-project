"""Reproducible known octahedral embeddings, not saved retraction maps.

A seeded target supplies only the location of an induced octahedron.
The retraction itself must still be found and verified by the solver.

The snub O24 coordinates use bitmask-ordered labels of the iterated
clique graphs of the specifically labeled F_{4,4}. Transport the seed
through an isomorphism of the eight-vertex source graph and through
the clique operator; avoid a costly K4 graph-isomorphism search.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import networkx as nx

# Antipodal pairs in bitmask-labeled K^4(F_{4,4}).
# These pairs are NOT proof: no retraction images are stored here.
_SNUB_O24_PAIRS: tuple[tuple[int, int], ...] = (
    (107, 1050),
    (366, 798),
    (303, 463),
    (28, 945),
    (101, 839),
    (510, 872),
    (298, 495),
    (301, 918),
    (265, 842),
    (355, 461),
    (262, 845),
    (260, 805),
    (111, 1066),
    (248, 1026),
    (744, 803),
    (522, 871),
    (476, 596),
    (321, 467),
    (573, 828),
    (474, 900),
    (245, 834),
    (318, 452),
    (247, 830),
    (309, 456),
)

_SNUB_ORDERS = (8, 12, 20, 56, 1076)


def _labeled_f44() -> nx.Graph:
    """Return the specific F_{4,4} labeling used for the O24 witness."""
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from((i, i + 1) for i in range(3))
    graph.add_edges_from((i, i + 1) for i in range(4, 7))
    graph.add_edges_from(
        (i, 4 + j) for i in range(4) for j in range(4) if i in (0, 3) or j in (0, 3)
    )
    return graph


def snub_o24_matches_source(graph: nx.Graph) -> bool:
    """Whether the given eight-vertex graph is a snub disphenoid."""
    return (
        graph.number_of_nodes() == 8
        and graph.number_of_edges() == 18
        and bool(nx.is_isomorphic(_labeled_f44(), graph))
    )


def snub_o24_pairs_for_iterates(
    iterates: Sequence[nx.Graph],
) -> tuple[tuple[Hashable, Hashable], ...] | None:
    """Locate the certified embedding in a pared snub clique sequence.

    iterates contains G, K(G), ..., K^4(G), in the original nested-clique
    NetworkX labeling. Works for any relabeling of the snub, including
    graph6 GQyuzw. Returns None if sequence orders/source differ.
    This does not read retraction images or run a retraction solver.
    """
    if len(iterates) != 5 or tuple(len(g) for g in iterates) != _SNUB_ORDERS:
        return None

    original = iterates[0]
    if original.number_of_edges() != 18:
        return None
    matcher = nx.isomorphism.GraphMatcher(_labeled_f44(), original)
    mapping: dict[int, Hashable] | None = next(matcher.isomorphisms_iter(), None)
    if mapping is None:
        return None

    for graph in iterates[1:]:
        old_indices = {vertex: index for index, vertex in mapping.items()}
        bitmasks: list[tuple[int, Hashable]] = []
        try:
            for clique in graph:
                if not isinstance(clique, frozenset):
                    return None
                mask = sum(1 << old_indices[v] for v in clique)
                bitmasks.append((mask, clique))
        except (KeyError, TypeError):
            return None
        # The source indices are distinct and frozensets are unique graph
        # vertices, so the subset-to-bitmask encoding is injective.
        bitmasks.sort(key=lambda pair: pair[0])
        mapping = {index: clique for index, (_, clique) in enumerate(bitmasks)}

    return tuple((mapping[a], mapping[b]) for a, b in _SNUB_O24_PAIRS)
