"""Reproduce a blind K^4(snub disphenoid) -> O24 retraction via public API.

No target coordinates, stored projection, or intermediate checkpoint are read.
The fixed fingerprint describes *only* the graph labeling and vertex order.

Usage: uv run python research/snub_blind_o24_from_api.py --output snub_blind.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import pycliques

EXPECTED_FINGERPRINT = (
    "0a892f39bb1201d722d19a884b117164b8698a540a6cc1ed6daf99bf802068da"
)


def snub_fourth_iterate() -> nx.Graph:
    """Reconstruct canonical F_{4,4} iterates using only public pycliques."""
    g = nx.Graph()
    g.add_nodes_from(range(8))
    g.add_edges_from((i, i + 1) for i in range(3))
    g.add_edges_from((i, i + 1) for i in range(4, 7))
    g.add_edges_from(
        (i, 4 + j)
        for i in range(4)
        for j in range(4)
        if i in (0, 3) or j in (0, 3)
    )
    for _ in range(4):
        h = pycliques.clique_graph(g, bound=1200)
        if h is None:
            raise AssertionError("clique-graph construction exceeded the bound")
        by_set = {frozenset(c): c for c in h}
        enumeration = [frozenset(c) for c in nx.find_cliques(g)]
        if len(by_set) != len(enumeration):
            raise AssertionError("independent maximal-clique counts disagree")
        canonical = sorted(by_set, key=lambda c: sum(1 << int(v) for v in c))
        numbering = {c: i for i, c in enumerate(canonical)}
        next_graph = nx.Graph()
        next_graph.add_nodes_from(numbering[c] for c in enumeration)
        for i, c in enumerate(enumeration):
            for d in enumeration[:i]:
                if h.has_edge(by_set[c], by_set[d]):
                    next_graph.add_edge(numbering[c], numbering[d])
        g = next_graph
    if (len(g), g.number_of_edges()) != (1076, 504360):
        raise AssertionError("unexpected K4 snub order or size")
    return g


def ordered_fingerprint(g: nx.Graph) -> str:
    digest = hashlib.sha256()
    for v in g:
        digest.update(f"{v},".encode())
    digest.update(b"|")
    for u, v in sorted((min(a, b), max(a, b)) for a, b in g.edges):
        digest.update(f"{u}:{v},".encode())
    return digest.hexdigest()


def reproduce(*, output: str | Path) -> dict:
    g = snub_fourth_iterate()
    fingerprint = ordered_fingerprint(g)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise AssertionError("canonical iteration order changed; investigate")
    result = pycliques.find_blind_octahedral_retraction(
        g,
        target_dimension=24,
        seed=24002,
        beam=12,
        samples=25,
        biclique_variants=1,
        core_rounds=1000,
        delete_count=7,
        max_repairs=10,
        completion_states=200000,
        retraction_states=200000,
        retraction_seed=9883,
    )
    if result.status is not pycliques.OctahedralBlindStatus.FOUND:
        raise RuntimeError("inconclusive blind search; no certificate produced")
    pairs, projection = result.antipodal_pairs, result.projection
    assert pairs is not None and projection is not None
    anchors = {v for pair in pairs for v in pair}
    target = g.subgraph(anchors)
    if not pycliques.verify_induced_octahedron(
        g, pairs, require_dominating=True
    ):
        raise AssertionError("octahedral embedding failed validation")
    if not pycliques.verify_retraction_witness(
        g, target, (projection, {v: v for v in anchors})
    ):
        raise AssertionError("public retraction verifier rejected the witness")
    if not (
        len(projection) == len(g)
        and all(projection[v] == v for v in anchors)
        and all(
            projection[u] == projection[v]
            or g.has_edge(projection[u], projection[v])
            for u, v in g.edges
        )
    ):
        raise AssertionError("independent all-edges verification failed")
    certificate = {
        "method": "blind conflict-guided octahedral retraction",
        "source": "K^4(F_4,4)",
        "fingerprint": fingerprint,
        "source_order": len(g),
        "source_edges": g.number_of_edges(),
        "partial_dimension": result.partial_dimension,
        "conflicts_learned": result.conflicts_learned,
        "repairs_considered": result.repairs_considered,
        "completion_states": result.completion_states,
        "target_pairs": [list(p) for p in pairs],
        "vertex_order": list(g),
        "images": [projection[v] for v in g],
    }
    Path(output).write_text(json.dumps(certificate, indent=2) + "\n")
    return certificate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="snub_blind_o24_reproduction.json")
    args = parser.parse_args()
    result = reproduce(output=args.output)
    print(
        "CERTIFIED: K4(snub) -> O24,",
        result["source_order"], "images,",
        result["source_edges"], "edges;",
        "output:", args.output,
    )
