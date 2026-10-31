#!/usr/bin/env python
"""Explore the proper part of the partition lattice Pi_n.

Given n, build the comparability graph of the nontrivial set partitions of
{0, ..., n-1} (all partitions except the finest and the coarsest), where two
partitions are adjacent when one refines the other. Then take its clique
complex (the order complex of the proper part of Pi_n) and decompose the
S_n-module H_r(K; C) into irreducibles (Specht modules).

Run with ``python -m research.partition_lattice N [--max-dim R]``.

Homology is the *unreduced* homology computed by pyhomrep, so H_0 includes
the trivial module when the complex is non-empty. For n >= 4, the top
homology H_{n-3} has dimension (n-1)!. For n = 3, the proper part consists
of three incomparable partitions: H_0 has dimension 3, while reduced H_0
has dimension 2 = (n-1)!.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from itertools import combinations

import networkx as nx
from pycombtop import SimplicialComplex, clique_complex
from pyhomrep import (
    character_p_homology,
    form_matrix_yt,
    make_permutation,
    partitions_list,
    size_conjugacy_class,
)
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import multiset_partitions

Partition = frozenset[frozenset[int]]
RepresentationDecomposition = dict[int, dict[tuple[int, ...], object]]


def nontrivial_partitions(n: int) -> list[Partition]:
    """Return partitions of ``{0, ..., n-1}`` with between 2 and n-1 blocks."""
    if n < 3:
        raise ValueError("n must be at least 3")
    return [
        frozenset(frozenset(b) for b in p)
        for p in multiset_partitions(list(range(n)))
        if 1 < len(p) < n
    ]


def refines(a: Partition, b: Partition) -> bool:
    """True if every block of a lies inside a block of b."""
    return all(any(x <= y for y in b) for x in a)


def partition_graph(n: int) -> tuple[nx.Graph, list[Partition]]:
    """Return the comparability graph and its indexed nontrivial partitions."""
    if n < 3:
        raise ValueError("n must be at least 3")
    parts = nontrivial_partitions(n)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(parts)))
    for i, j in combinations(range(len(parts)), 2):
        if refines(parts[i], parts[j]) or refines(parts[j], parts[i]):
            graph.add_edge(i, j)
    return graph, parts


def induced_permutation(
    g: Permutation, n: int, parts: list[Partition], index: dict[Partition, int]
) -> Permutation:
    """Return the permutation of partition indices induced by ``g`` in ``S_n``."""
    g = Permutation(g.array_form, size=n)
    img = [
        index[frozenset(frozenset(g(x) for x in block) for block in p)] for p in parts
    ]
    return Permutation(img, size=len(parts))


def decompose(
    n: int, max_dim: int | None = None, verbose: bool = True
) -> RepresentationDecomposition:
    """Decompose each computed homology representation into Specht modules.

    Args:
        n: Size of the underlying set; must be at least three.
        max_dim: Optional nonnegative upper bound on the homology degrees.
        verbose: Print graph and degree summaries when true.

    Returns:
        A mapping from homology degree to Specht multiplicities.
    """
    if n < 3:
        raise ValueError("n must be at least 3")
    if max_dim is not None and max_dim < 0:
        raise ValueError("max_dim must be nonnegative or None")

    graph, parts = partition_graph(n)
    index = {p: i for i, p in enumerate(parts)}
    complex_: SimplicialComplex = clique_complex(graph)
    top = complex_.dimension()
    if max_dim is not None:
        top = min(top, max_dim)
    if verbose:
        print(
            f"n={n}: {graph.number_of_nodes()} vertices, "
            f"{graph.number_of_edges()} edges, dim K = {complex_.dimension()}"
        )

    classes = partitions_list(n)
    chi = form_matrix_yt(classes)
    sizes = [size_conjugacy_class(c, n) for c in classes]
    reps = [induced_permutation(make_permutation(c), n, parts, index) for c in classes]

    result: RepresentationDecomposition = {}
    for r in range(top + 1):
        chars = [character_p_homology(complex_, r, g) for g in reps]
        dim = chars[classes.index([1] * n)]
        mult = {}
        for i, lam in enumerate(classes):
            total = sum(chi[i, j] * chars[j] * sizes[j] for j in range(len(classes)))
            m = total / math.factorial(n)
            if m != 0:
                mult[tuple(lam)] = m
        result[r] = mult
        if verbose:
            print(
                f"H_{r}: dim {dim}  =  "
                + (" + ".join(f"{m}*S{lam}" for lam, m in mult.items()) or "0")
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the partition-lattice decomposition command-line interface."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", type=int, help="size of the underlying set")
    ap.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="only compute homology up to this degree",
    )
    args = ap.parse_args(argv)
    if args.n < 3:
        ap.error("n must be at least 3 (otherwise the graph is empty)")
    if args.max_dim is not None and args.max_dim < 0:
        ap.error("--max-dim must be nonnegative")
    decompose(args.n, args.max_dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
