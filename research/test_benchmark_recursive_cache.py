"""Regression test: all-timeout samples must still produce a benchmark artifact."""

import json
import sys

from research import benchmark_recursive_cache as bench


def test_all_timeout_cases_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "SAMPLE", (("example", "A?", "synthetic timeout"),))
    monkeypatch.setattr(
        bench,
        "measure",
        lambda *args, **kwargs: {
            "status": "timeout",
            "seconds": 0.001,
            "cache_stats": {},
        },
    )
    filename = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--repeats", "2", "--json-out", str(filename)],
    )
    bench.main()
    data = json.loads(filename.read_text(encoding="utf-8"))
    assert len(data["results"]) == 1
    row = data["results"][0]
    assert row["fully_comparable"] is False
    assert row["ratio_old_over_new"] is None
    assert len(row["runs"]["uncached"]) == 2
    assert len(row["runs"]["cached"]) == 2
