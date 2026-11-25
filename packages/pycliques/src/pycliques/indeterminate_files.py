"""Read and update historical small-graph census rows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import networkx as nx

from .classification_types import Certificate

_logger = logging.getLogger("pycliques.small")


def _indeterminate_file_path(order: int, data_dir: Path) -> Path:
    """Return the path for the indeterminate-graphs file of a given order."""
    return data_dir / f"indeterminate_order_{order}.txt"


def _serialize_certificate(certificate: Certificate | None) -> str:
    """Render certificate metadata for the indeterminate-graphs file."""
    if certificate is None:
        return "- - -"
    target_label = certificate.target_label or "-"
    target_status = certificate.target_status or "-"
    return f"{certificate.rule} {target_status} {target_label}"


def _save_indeterminate(
    order: int,
    indeterminate: list[tuple[int, nx.Graph, Certificate | None]],
    data_dir: Path,
    *,
    replace: bool = False,
) -> None:
    """Save indeterminate pared graphs to a human-readable file.

    Each line contains the original graph index, the order of the pared
    graph, its graph6 string, and certificate metadata when available.

    If the file already exists, the new results are merged with the
    existing entries.  Duplicate indices are resolved in favour of the
    new run so that re-processing a range always updates the record.  When
    ``replace`` is true, the file is replaced by exactly the supplied
    indeterminate entries.
    """
    path = _indeterminate_file_path(order, data_dir)

    # Load existing entries keyed by original index.
    existing: dict[int, str] = {}
    if path.is_file() and not replace:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split(maxsplit=5)
                existing[int(parts[0])] = stripped
        _logger.info(
            f"Found existing file with {len(existing)} entries; merging new results"
        )

    # Build lines for the new entries (overwrite any duplicate index).
    for idx, graph, certificate in indeterminate:
        g = nx.convert_node_labels_to_integers(graph)
        g6 = nx.to_graph6_bytes(g, header=False).decode("ascii").strip()
        existing[idx] = f"{idx} {g.order()} {g6} {_serialize_certificate(certificate)}"

    # Write everything back sorted by index.
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary_path = Path(f.name)
            f.write(
                f"# Indeterminate clique behavior - connected graphs of order {order}\n"
            )
            f.write(
                "# Format: original_index pared_order graph6 certificate_rule "
                "certificate_target_status certificate_target_label\n"
            )
            for idx in sorted(existing):
                f.write(f"{existing[idx]}\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    _logger.info(f"Saved {len(existing)} indeterminate graphs to {path}")


def _update_processed_indeterminate(
    order: int,
    processed_indices: set[int],
    indeterminate: list[tuple[int, nx.Graph, Certificate | None]],
    data_dir: Path,
) -> None:
    """Replace unresolved rows for processed indices, preserving all others."""
    existing = _load_indeterminate_file_entries_with_metadata(order, data_dir)
    previous_by_index = {
        index: (graph, certificate) for index, graph, certificate in existing
    }
    retained = [entry for entry in existing if entry[0] not in processed_indices]
    refreshed = []
    for index, graph, certificate in indeterminate:
        previous = previous_by_index.get(index)
        if (
            certificate is None
            and previous is not None
            and previous[1] is not None
            and nx.is_isomorphic(graph, previous[0])
        ):
            certificate = previous[1]
        refreshed.append((index, graph, certificate))
    _save_indeterminate(
        order,
        retained + refreshed,
        data_dir,
        replace=True,
    )


def _graph6_string(graph: nx.Graph) -> str:
    """Return a graph6 string after normalizing arbitrary node labels."""
    numbered = nx.convert_node_labels_to_integers(graph)
    return cast(str, nx.to_graph6_bytes(numbered, header=False).decode("ascii").strip())


def _load_indeterminate_graphs(
    max_order: int,
    data_dir: Path,
) -> dict[int, list[nx.Graph]]:
    """Load indeterminate graphs from files for all orders less than *max_order*.

    .. rubric:: Returns

    dict[int, list[nx.Graph]]
        Graphs grouped by their vertex count so that lookups only need to
        test isomorphism against candidates of the same size.
    """
    by_vertex_count: dict[int, list[nx.Graph]] = {}
    for order in range(1, max_order):
        path = _indeterminate_file_path(order, data_dir)
        if not path.is_file():
            continue
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=5)
                g6_str = parts[2]
                graph = nx.from_graph6_bytes(g6_str.encode("ascii"))
                n = graph.order()
                by_vertex_count.setdefault(n, []).append(graph)
                count += 1
        _logger.info(f"Loaded {count} indeterminate graphs from order-{order} file")
    return by_vertex_count


def _is_known_indeterminate(
    graph: nx.Graph,
    known: dict[int, list[nx.Graph]],
) -> bool:
    """Return whether *graph* is isomorphic to any known indeterminate graph."""
    candidates = known.get(graph.order(), [])
    return any(nx.is_isomorphic(graph, c) for c in candidates)


def _load_indeterminate_file_entries(
    order: int, data_dir: Path
) -> list[tuple[int, nx.Graph]]:
    """Load ``(index, graph)`` pairs previously saved for *order* by the fast pass."""
    return [
        (index, graph)
        for index, graph, _certificate in (
            _load_indeterminate_file_entries_with_metadata(order, data_dir)
        )
    ]


def _load_indeterminate_file_entries_with_metadata(
    order: int, data_dir: Path
) -> list[tuple[int, nx.Graph, Certificate | None]]:
    """Load saved indeterminate entries together with certificate metadata."""
    path = _indeterminate_file_path(order, data_dir)
    entries: list[tuple[int, nx.Graph, Certificate | None]] = []
    if not path.is_file():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=5)
            certificate = None
            if len(parts) >= 6 and parts[3] != "-":
                certificate = Certificate(
                    rule=parts[3],
                    target_status=None if parts[4] == "-" else parts[4],
                    target_label=None if parts[5] == "-" else parts[5],
                )
            entries.append(
                (
                    int(parts[0]),
                    nx.from_graph6_bytes(parts[2].encode("ascii")),
                    certificate,
                )
            )
    return entries
