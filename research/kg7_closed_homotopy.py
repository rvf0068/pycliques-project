"""Reproduce the KG(7,2) closed-complete tests.

Run with uv run python -m research.kg7_closed_homotopy --max-checks 100000.
Labels are sorted maximal cliques at each iteration. The S7 action is lifted
using adjacent transpositions of the seven points. The index verifies that
these permutations preserve the family incidence before orbit reduction.
"""

import argparse
from itertools import combinations

import networkx as nx

from pycliques import ClosedNecktieIndex, check_closed_centers, clique_graph
from pycombtop import check_closed_trace_contractibility
from pycombtop.nerve_reduction import (
    alternating_nerve_reduction,
    verify_nerve_reduction,
)


def family_and_generators(iterate: int):
    edges = list(combinations(range(7), 2))
    edge_index = {e: i for i, e in enumerate(edges)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(edges)))
    graph.add_edges_from(
        (i, j)
        for i, j in combinations(graph, 2)
        if not set(edges[i]) & set(edges[j])
    )
    generators = []
    for k in range(6):
        p = list(range(7))
        p[k], p[k + 1] = p[k + 1], p[k]
        generators.append(
            [edge_index[tuple(sorted(p[v] for v in e))] for e in edges]
        )
    for step in range(iterate + 1):
        # At the last step only the maximal cliques are needed.
        family = sorted(
            map(frozenset, nx.find_cliques(graph)), key=lambda q: tuple(sorted(q))
        )
        lookup = {q: j for j, q in enumerate(family)}
        generators = [
            [lookup[frozenset(p[v] for v in q)] for q in family]
            for p in generators
        ]
        if step != iterate:
            next_graph = clique_graph(graph)
            assert next_graph is not None
            graph = nx.relabel_nodes(next_graph, lookup)
    return family, generators


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-checks", type=int, default=100_000)
    parser.add_argument("--iterate", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--strategy", choices=["maximal", "vertex"], default="maximal")
    args = parser.parse_args()
    if args.max_checks < 0:
        parser.error("--max-checks must be nonnegative")
    if args.iterate == 3 and not (args.lazy and args.uniform and args.strategy == "vertex"):
        parser.error("iterate 3 requires --lazy --uniform --strategy vertex")
    family, generators = family_and_generators(args.iterate)
    index = ClosedNecktieIndex(family, lazy=args.lazy)
    if args.uniform:
        result = check_closed_trace_contractibility(
            index,
            generators=generators,
            max_nodes=args.max_checks,
            strategy=args.strategy,
        )
        print(result, flush=True)
        if result.proved:
            print("PROVED: LPV08 Theorem 5.7 contractibility hypothesis")
        else:
            print("INCONCLUSIVE: incomplete enumeration or unresolved local complex")
        return
    centers = check_closed_centers(
        index, generators=generators, max_checks=args.max_checks
    )
    print("Center criterion:", centers, flush=True)
    if centers.holds is False:
        print(
            "Center criterion fails; Theorem 5.7 contractibility test is independent.",
            flush=True,
        )
    if centers.witness is not None:
        print(
            "center witness indices:",
            [i for i in range(len(family)) if centers.witness & (1 << i)],
            flush=True,
        )
        reduction = alternating_nerve_reduction(index.nerve_facets(centers.witness))
        assert verify_nerve_reduction(index.nerve_facets(centers.witness), reduction)
        print("witness reduction:", reduction.facet_counts, flush=True)

    checked = bad = unknown = 0
    exhaustive = True
    for mask in index.closed_representatives(generators):
        if checked >= args.max_checks:
            exhaustive = False
            break
        checked += 1
        if index.is_bad(mask):
            bad += 1
            facets = index.nerve_facets(mask)
            reduction = alternating_nerve_reduction(facets)
            if args.verify:
                assert verify_nerve_reduction(facets, reduction)
            if not reduction.proved_contractible:
                unknown += 1
                print("unresolved closed complete:", mask, flush=True)
        if checked % 10_000 == 0:
            print("checked", checked, "bad", bad, "unknown", unknown, flush=True)
    print("checked", checked, "bad", bad, "unknown", unknown, flush=True)
    # LPV08 Theorem 5.7 requires contractibility of Delta(h(A)), not
    # a center. Good closed completes have a common vertex and hence a
    # simplex nerve; all bad closed completes must have passed the reduction above.
    # A failed center test does not refute this weaker sufficient condition.
    if exhaustive and not unknown:
        print("PROVED: LPV08 Theorem 5.7 contractibility hypothesis")
    else:
        print("INCONCLUSIVE: incomplete enumeration or unresolved local complex")


if __name__ == "__main__":
    main()
