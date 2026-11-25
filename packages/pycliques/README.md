# pycliques

Compute the clique graph and its raw or pared iterates, classify clique behavior, and search for graph retractions. From the repository root, install with `pip install ./packages/pycliques`, or run `uv sync` for the workspace.

```python
import networkx as nx
from pycliques import CliqueSequence, classify_clique_behavior

graph = nx.cycle_graph(5)
sequence = CliqueSequence(graph, pare=False)
assert len(sequence[1]) == 5
print(classify_clique_behavior(graph).verdict)
```

Install the sibling `pyg6data` wheel for census data, `pycombtop` for functions returning simplicial complexes, and `pycliques[milp]` for the optional SciPy solver. [Methods](https://github.com/rvf0068/pycliques-project/blob/main/docs/methods.rst) · [Classifier and examples](https://github.com/rvf0068/pycliques-project/blob/main/docs/clique_behavior.md)
