[![CI](https://github.com/rvf0068/pycliques-project/actions/workflows/ci.yml/badge.svg)](https://github.com/rvf0068/pycliques-project/actions/workflows/ci.yml) [![Lint](https://github.com/rvf0068/pycliques-project/actions/workflows/lint.yml/badge.svg)](https://github.com/rvf0068/pycliques-project/actions/workflows/lint.yml) [![codecov](https://codecov.io/gh/rvf0068/pycliques-project/branch/main/graph/badge.svg)](https://codecov.io/gh/rvf0068/pycliques-project)

# pycliques-project

Research software for iterated clique graphs, simplicial complexes and group actions on homology. Throughout the workspace a **clique** is an inclusion-maximal nonempty complete.

| Package | What it provides |
|---|---|
| [pycliques](packages/pycliques) | Clique iteration, behavior classification, retractions and replayable divergence witnesses |
| [pycombtop](packages/pycombtop) | Simplicial complexes, homotopy certificates, homology and covering graphs |
| [pyg6data](packages/pyg6data) | Indexed graph6 datasets, directed graphs and resumable census runs |
| [pyhomrep](packages/pyhomrep) | Representations of groups acting on simplicial homology |

## Getting started

All packages live under `packages/` in this repository. Clone and install the ones you need:

```bash
git clone https://github.com/rvf0068/pycliques-project.git
cd pycliques-project
```

**With uv** (recommended — resolves the sibling packages from this checkout):

```bash
uv sync                        # set up the cloned workspace (all packages + dev tools)
```

Use `uv add pycliques` only to add the published package to a *different*
project; it edits that project's dependencies and is not a setup step here.

**With pip** (install individual packages from this checkout). Install
`pyg6data` before packages that depend on it, and use local paths so pip does
not resolve sibling packages from PyPI:

```bash
pip install ./packages/pyg6data
pip install ./packages/pycliques
pip install ./packages/pycombtop
pip install ./packages/pyhomrep
```

## Working with the repository

Run `make test` for package, doctest and fast research coverage, or `make research-test` for the research tests. The ordinary suite excludes census-scale replays marked `slow`. The dedicated workflows carry out the native and MILP checks.

The [method guide](docs/methods.rst) links the octahedral searches and explains the evidence each one returns. The [clique behavior guide](docs/clique_behavior.md) and [certified homotopy reductions](docs/certified_homotopy.md) include extended examples and the mathematical arguments. The [references](docs/references.rst) give the original sources.
