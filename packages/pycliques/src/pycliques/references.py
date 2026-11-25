"""Divergent reference graphs and scoped selections for recursive classifiers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import cast

import networkx as nx

from .classification_types import ReferenceStatus
from .named import snub_disphenoid

ReferenceFact = tuple[nx.Graph, ReferenceStatus, str | None]
ReferenceSelection = tuple[ReferenceFact, ...]
_active: ContextVar[ReferenceSelection | None] = ContextVar(
    "pycliques_references", default=None
)

_REFERENCE_GRAPHS: dict[bytes, tuple[nx.Graph, ReferenceStatus, str | None]] = {}


def _canonical_reference_key(graph: nx.Graph) -> bytes:
    """Encode the current vertex order for use as a storage key."""
    numbered = nx.convert_node_labels_to_integers(graph)
    return cast(bytes, nx.to_graph6_bytes(numbered, header=False))


def register_reference_graph(
    graph: nx.Graph,
    *,
    status: ReferenceStatus = ReferenceStatus.PROVEN,
    label: str | None = None,
) -> None:
    """Register a graph whose clique behavior is known independently of the classifier.

    The status distinguishes a proven divergent reference graph from a merely
    conjectured one. The classifier uses the same inference rules for every
    registered graph, so future reference facts can be added without branching
    on individual graph names.

    label : str, optional
        Human-readable identifier for the reference graph, used in certificates
        and saved indeterminate-file metadata.
    """
    key = _canonical_reference_key(graph)
    for existing, (known, previous, _) in _REFERENCE_GRAPHS.items():
        if (len(known), known.size()) == (
            len(graph),
            graph.size(),
        ) and nx.is_isomorphic(known, graph):
            if (
                previous is ReferenceStatus.PROVEN
                and status is ReferenceStatus.CONJECTURED
            ):
                return
            key = existing
            break
    _REFERENCE_GRAPHS[key] = (nx.freeze(graph.copy()), status, label)


def clear_reference_graphs() -> None:
    """Clear all registered divergent reference graphs."""
    _REFERENCE_GRAPHS.clear()


register_reference_graph(
    snub_disphenoid(),
    status=ReferenceStatus.PROVEN,
    label="snub_disphenoid",
)

register_reference_graph(
    nx.octahedral_graph(),
    status=ReferenceStatus.PROVEN,
    label="octahedral_graph",
)


def select_reference_graphs(*, proved_only: bool = False) -> ReferenceSelection:
    """Snapshot the registry, optionally excluding conjectured divergence."""
    return tuple(
        fact
        for fact in _REFERENCE_GRAPHS.values()
        if not proved_only or fact[1] is ReferenceStatus.PROVEN
    )


def current_references(defaults: Iterable[ReferenceFact]) -> ReferenceSelection:
    active = _active.get()
    return tuple(defaults) if active is None else active


@contextmanager
def reference_scope(selection: ReferenceSelection) -> Iterator[None]:
    token = _active.set(selection)
    try:
        yield
    finally:
        _active.reset(token)
