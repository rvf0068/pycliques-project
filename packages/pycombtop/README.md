## Indeterminate graph files

Compute the homotopy type of each graph's clique complex and write a tab-separated
report alongside the input file:

```bash
uv run homotopy-indeterminate indeterminate_order_9.txt
```

The default output is `indeterminate_order_9_homotopy.txt`. Choose another path
with `--output`; use `--max-vertices` or `--max-simplices` to limit expensive
computations. The report preserves each graph's index, order, and graph6 string.
Its status is `exact`, `betti-only`, or `limited`, so inconclusive results are
not presented as proven homotopy types.
