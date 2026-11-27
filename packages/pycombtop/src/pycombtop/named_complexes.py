"""Named simplicial complexes for experiments and regression examples."""

import math
from collections.abc import Iterable, Sequence

import networkx as nx
from pycliques.named import matching_graph, octahedron

from .simplex import SimplicialComplex, clique_complex


def _nonnegative(n: int, name: str) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def independence_complex(graph: nx.Graph) -> SimplicialComplex:
    """Complex of independent vertex sets of a finite simple graph."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("expected a simple undirected graph")
    return clique_complex(nx.complement(graph))


def matching_complex(graph: nx.Graph | int) -> SimplicialComplex:
    """Complex of matchings of a simple graph; an integer denotes K_n."""
    return clique_complex(matching_graph(graph))


def chessboard_complex(m: int, n: int) -> SimplicialComplex:
    """Nonattacking rook placements on an m by n board, labeled by (row, column)."""
    _nonnegative(m, "m")
    _nonnegative(n, "n")
    graph = nx.Graph()
    graph.add_nodes_from((i, j) for i in range(m) for j in range(n))
    graph.add_edges_from(
        ((i, j), (k, column))
        for i in range(m)
        for k in range(i + 1, m)
        for j in range(n)
        for column in range(n)
        if j != column
    )
    return clique_complex(graph)


def simplex_boundary(n: int) -> SimplicialComplex:
    """Boundary of the simplex on n vertices, of dimension n-2 for n >= 1."""
    _nonnegative(n, "n")
    vertices = set(range(n))
    return SimplicialComplex(vertices, facet_set=(vertices - {v} for v in vertices))


def cross_polytope_boundary(n: int) -> SimplicialComplex:
    """Boundary of the n-dimensional cross-polytope, a flag (n-1)-sphere."""
    _nonnegative(n, "n")
    if n == 0:
        return SimplicialComplex(set(), facet_set=[set()])
    return clique_complex(octahedron(n))


def sphere_complex(d: int) -> SimplicialComplex:
    """Standard d-sphere as the boundary of a (d+1)-simplex."""
    if not isinstance(d, int) or isinstance(d, bool) or d < -1:
        raise ValueError("d must be an integer at least -1")
    return simplex_boundary(d + 2)


def projective_plane_complex() -> SimplicialComplex:
    """The six-vertex triangulation of the real projective plane."""
    triangles = [
        (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
        (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
    ]
    return SimplicialComplex(range(6), facet_set=triangles)


def _grid_surface(*, twist: bool) -> SimplicialComplex:
    """Triangulate a square grid with periodic or twisted identifications."""
    n = 4

    def vertex(i: int, j: int) -> tuple[int, int]:
        if j == n and twist:
            i = -i
        return i % n, j % n

    facets = []
    for i in range(n):
        for j in range(n):
            a, b = vertex(i, j), vertex(i + 1, j)
            c, d = vertex(i, j + 1), vertex(i + 1, j + 1)
            facets.extend([(a, b, d), (a, c, d)])
    return SimplicialComplex(
        ((i, j) for i in range(n) for j in range(n)), facet_set=facets
    )


def torus_complex() -> SimplicialComplex:
    """A 16-vertex triangulation of the torus from a periodic square grid."""
    return _grid_surface(twist=False)


def klein_bottle_complex() -> SimplicialComplex:
    """A 16-vertex triangulation of the Klein bottle from a twisted grid."""
    return _grid_surface(twist=True)


def vietoris_rips_complex(
    points: Iterable[Sequence[float]], epsilon: float
) -> SimplicialComplex:
    """Euclidean Vietoris-Rips complex at an inclusive distance threshold.

    Vertices are point indices, so repeated coordinates remain distinct.
    This explicit complex can be large; use GUDHI for large filtrations.
    """
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    coordinates = [tuple(point) for point in points]
    if coordinates and any(
        len(point) != len(coordinates[0]) or not all(map(math.isfinite, point))
        for point in coordinates
    ):
        raise ValueError("points must have the same dimension and finite coordinates")
    graph = nx.Graph()
    graph.add_nodes_from(range(len(coordinates)))
    for i, a in enumerate(coordinates):
        for j in range(i + 1, len(coordinates)):
            if math.dist(a, coordinates[j]) <= epsilon:
                graph.add_edge(i, j)
    return clique_complex(graph)
