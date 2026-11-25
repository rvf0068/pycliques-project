# pyg6data

Read the bundled McKay graphs and digraphs, House of Graphs connected cubic graphs, and small torsion examples. From the repository root run `pip install ./packages/pyg6data` or `uv sync`.

```python
from pathlib import Path
from pyg6data import iter_graph6_records, iter_graph_census

index, encoding = next(iter_graph6_records(10))
rows = iter_graph_census(6, lambda graph: True, lambda graph: graph.number_of_edges(),
                        checkpoint=Path("edges.sqlite"), run_id="edge-count-v1")
for row in rows:
    print(row["index"], row["result"])
```

Indices are zero-based in bundled file order. Parallel runs require picklable callables. Use `run_graph_census` to collect the same rows into a list. A checkpoint records the source digest, order and run identifier. Change the identifier when changing the filter, evaluator or settings. To import an old JSON file, call `import_legacy_checkpoint(old_path, new_path, order=6, run_id="edge-count-v1")` with its original calculation identity.
