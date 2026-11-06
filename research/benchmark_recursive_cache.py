"""Controlled same-revision before/after benchmark for recursive memoization.

Run:
    uv run python research/benchmark_recursive_cache.py --repeats 2 --timeout 12

The uncached mode disables only recursive memoization: theorem tests,
parameters, implementation version, and sample graphs are identical.
No result is treated as a mathematical proof without its certificate.
"""
from __future__ import annotations

import argparse
import json
import platform
import signal
import statistics
import time
from pathlib import Path

import networkx as nx

from pycliques.small import classify_clique_behavior


# Fixed graph6 examples from the completed order-ten, bound120-timeboxed-v2
# census. Exactly this list and its order are reproduced in the JSON output.
SAMPLE: tuple[tuple[str, str, str], ...] = (
    ("11496773", "ICvbvy}|g", "convergent"),
    ("11621894", "IEivrzm}o", "convergent"),
    ("11708244", "IQyuvh}xo", "retraction divergence"),
    ("11634010", "IEjer^mvO", "special octahedron at K2"),
    ("11716486", "IUzvvz}^o", "high-dimension special octahedron"),
    ("10294355", "ICZJb^Un_", "special octahedron at K3"),
    ("11590312", "IEhuuZxNo", "bound exceeded"),
    ("11385189", "ICrbTvw[w", "recursive search cost"),
    ("11713770", "IQzTvzu]w", "historical special-octahedron timeout"),
)


class BenchmarkTimeout(Exception):
    """An individual case exceeded the configured wall-clock limit."""


def _expired(_signum: int, _frame: object) -> None:
    raise BenchmarkTimeout


def measure(
    graph6: str, *, cached: bool, timeout: float, bound: int, tries: int
) -> dict:
    graph = nx.from_graph6_bytes(graph6.encode("ascii"))
    stats: dict[str, int] = {}
    started = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        outcome = classify_clique_behavior(
            graph,
            bound=bound,
            tries=tries,
            retraction_depth=2,
            recursive_cache=cached,
            cache_stats=stats,
        )
        return {
            "status": "ok",
            "seconds": round(time.perf_counter() - started, 6),
            "verdict": outcome.verdict.name,
            "bound_exceeded": bool(outcome.bound_exceeded),
            "reason": outcome.reason,
            "cache_stats": stats,
        }
    except BenchmarkTimeout:
        return {
            "status": "timeout",
            "seconds": round(time.perf_counter() - started, 6),
            "cache_stats": stats,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--bound", type=int, default=120)
    parser.add_argument("--tries", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    signal.signal(signal.SIGALRM, _expired)
    rows = []
    print(
        "Index | Graph6 | Old/no cache (s) | New/cache (s) | "
        "Ratio old/new | Hits | Result"
    )
    print("--- | --- | ---: | ---: | ---: | ---: | ---", flush=True)
    for index, graph6, label in SAMPLE:
        raw = {"uncached": [], "cached": []}
        # Interleave modes per repetition to limit frequency/temperature bias.
        for _ in range(args.repeats):
            for cached, name in ((False, "uncached"), (True, "cached")):
                raw[name].append(
                    measure(
                        graph6,
                        cached=cached,
                        timeout=args.timeout,
                        bound=args.bound,
                        tries=args.tries,
                    )
                )
        old, new = raw["uncached"], raw["cached"]
        outcomes = [r for r in old + new if r["status"] == "ok"]
        # An entirely timed-out sample has no contradictory verdict; retain
        # its timeout records and emit a complete artifact.
        consistent = not outcomes or all(
            (r["verdict"], r["bound_exceeded"], r["reason"])
            == (
                outcomes[0]["verdict"],
                outcomes[0]["bound_exceeded"],
                outcomes[0]["reason"],
            )
            for r in outcomes
        )
        if not consistent:
            raise AssertionError(f"Classification changed for graph {index}")
        old_completed = [r["seconds"] for r in old if r["status"] == "ok"]
        new_completed = [r["seconds"] for r in new if r["status"] == "ok"]
        comparable = len(old_completed) == len(new_completed) == args.repeats
        old_median = statistics.median(old_completed) if old_completed else None
        new_median = statistics.median(new_completed) if new_completed else None
        ratio = old_median / new_median if comparable and new_median else None
        hits = sum(r["cache_stats"].get("hits", 0) for r in new)
        status = (
            outcomes[0]["verdict"] if outcomes and comparable else "timeout/partial"
        )
        rows.append(
            {
                "index": index,
                "graph6": graph6,
                "cohort_label": label,
                "runs": raw,
                "old_median_seconds": old_median,
                "new_median_seconds": new_median,
                "ratio_old_over_new": ratio,
                "cache_hits_total": hits,
                "fully_comparable": comparable,
            }
        )
        old_text = f"{old_median:.4f}" if old_median is not None else "timeout"
        new_text = f"{new_median:.4f}" if new_median is not None else "timeout"
        ratio_text = f"{ratio:.2f}x" if ratio is not None else "n/a"
        print(
            f"{index} | `{graph6}` | {old_text} | {new_text} | "
            f"{ratio_text} | {hits} | {status}",
            flush=True,
        )
    output = {
        "schema": 1,
        "purpose": "compare per-root recursive cache enabled vs disabled",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "networkx": nx.__version__,
        "parameters": {
            "repeats": args.repeats,
            "per_run_timeout": args.timeout,
            "bound": args.bound,
            "tries": args.tries,
            "retraction_depth": 2,
        },
        "results": rows,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
