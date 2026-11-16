"""Reproducible covering-morphism search on eight historically unresolved cores.

This is a research benchmark, NOT an additional classification stage. Inputs
are copied from the frozen nine-core cross-reference of pycliques-census,
reports/order10_core9_crossref_v1/matches.tsv (November 2026).
The historical 100,000-assignment searches all returned LIMIT_REACHED.
Successful certificates are replayed by the general library verifier.

Usage: uv run python research/benchmark_coaffine_morphism.py --output /tmp/covers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import networkx as nx
from pycliques.clockwork_search import CertificateVerificationStatus
from pycombtop.covering_divergence import (
    verify_triangular_cover_divergence_certificate,
)
from pycombtop.covering_divergence_search import search_covering_divergence

# Distinct completely pared nine-vertex core classes, not original
# ten-vertex census graphs. Historical order-nine source indices are
# included to prevent mixing the two index domains.
CASES = (
    (2317843, 180593, "HCRSv\\}"),
    (3148541, 220642, "HCY]Yzr"),
    (3505942, 220895, "HCZIt~u"),
    (4952669, 233406, "HCpumzi"),
    (4953338, 232908, "HCpunzi"),
    (5601875, 251445, "HCvVZpt"),
    (6938156, 251414, "HEhtnV{"),
    (7315343, 260038, "HElzvJb"),
)
# Same finite-family limits used by core9-followup-v1 in
# pycliques-census; only the general library search algorithm changes.
LIMITS = {
    "max_cover_classes": 1,
    "max_cover_vertices": 18,
    "max_cover_edges": 62,
    "max_clique_vertices": 350,
    "max_clique_edges": 6000,
    "max_source_order": 18,
    "max_m": 3,
    "max_n": 0,
    "max_morphism_states": 25000,
    "max_morphism_assignments": 100000,
}


def benchmark() -> dict:
    rows = []
    for core, order9, graph6 in CASES:
        graph = nx.from_graph6_bytes(graph6.encode("ascii"))
        if graph.order() != 9 or not nx.is_connected(graph):
            raise ValueError(f"Invalid graph6 for nine-core {core}")
        begin = perf_counter()
        result = search_covering_divergence(graph, **LIMITS)
        seconds = round(perf_counter() - begin, 4)
        certificate_verified = False
        if result.certificate is not None:
            replay = verify_triangular_cover_divergence_certificate(
                result.certificate
            )
            certificate_verified = replay.status is CertificateVerificationStatus.VERIFIED
            if not certificate_verified:
                raise AssertionError(f"Invalid new certificate at core {core}")
        entry = {
            "core_id": core,
            "order9_index": order9,
            "graph6": graph6,
            "status": result.status.name,
            "certificate_verified": certificate_verified,
            "morphism_states": result.morphism_states,
            "morphism_assignments": result.morphism_assignments,
            "cover_classes_total": result.cover_classes_total,
            "cover_classes_checked": result.cover_classes_checked,
            "morphisms_complete": result.morphisms_complete,
            "reasons": list(result.reasons),
            "elapsed_seconds": seconds,
        }
        rows.append(entry)
        print(
            f"core {core}: {entry['status']}, states={entry['morphism_states']}, "
            f"assignments={entry['morphism_assignments']}, {seconds:.3f}s",
            flush=True,
        )
    return {
        "schema": 1,
        "purpose": "algorithmic benchmark, not a new frozen census classification",
        "historical_assignments_per_core": 100000,
        "historical_status_per_core": "LIMIT_REACHED",
        "limits": LIMITS,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
