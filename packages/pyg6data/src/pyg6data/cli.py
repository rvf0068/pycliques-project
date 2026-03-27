"""Command-line interface for extracting graphs from g6 data files."""

import argparse
import gzip
import sys
from pathlib import Path

# Import the secure resource paths we built earlier
from .lists import _dict_connected, _get_data_file_path


def _parse_args(args=None) -> argparse.Namespace:
    """Parse command-line arguments for the extract-graphs command.

    .. rubric:: Parameters

    args : list[str] | None
        Argument list to parse (defaults to ``sys.argv[1:]``).

    .. rubric:: Returns

    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract specific graphs from a .g6.gz file by their index."
    )
    # Group inputs: You can either provide a local file OR ask for internal data
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-ff",
        "--from-file",
        help="Path to a custom .g6.gz file on your computer.",
        type=str,
    )
    group.add_argument(
        "-o",
        "--order",
        help="Order of connected graphs to extract from internal data (e.g., 6).",
        type=int,
    )

    # Required arguments
    parser.add_argument(
        "-l",
        "--list",
        nargs="+",
        type=int,
        required=True,
        help="Space-separated list of graph indices to extract (e.g., -l 2 10 200)",
    )
    parser.add_argument(
        "-tf",
        "--to-file",
        required=True,
        help="Output file path (e.g., -tf outfile.g6)",
        type=str,
    )
    return parser.parse_args(args)


def main(args=None):
    """Entry point for the ``extract-graphs`` command.

    Reads a ``.g6.gz`` file (internal or user-supplied), extracts the
    graphs at the requested indices, and writes them to an output file.

    .. rubric:: Parameters

    args : list[str] | None
        Argument list to parse (defaults to ``sys.argv[1:]``).
    """
    parsed_args = _parse_args(args)

    # 1. Resolve the source file
    if parsed_args.order:
        if parsed_args.order not in _dict_connected:
            print(f"Error: Internal data for order {parsed_args.order} not available.")
            sys.exit(1)
        # Use importlib.resources to securely fetch the internal data
        source_path = _get_data_file_path(_dict_connected[parsed_args.order])
    else:
        # Use the explicit file path provided by the user
        source_path = Path(parsed_args.from_file)
        if not source_path.exists():
            print(f"Error: File '{source_path}' does not exist.")
            sys.exit(1)

    # 2. Setup extraction parameters (FIXED VARIABLE NAMES HERE)
    indices_to_extract = set(parsed_args.list)
    max_index = max(indices_to_extract)
    extracted_count = 0

    print(f"Extracting {len(indices_to_extract)} graphs to '{parsed_args.to_file}'...")

    open_func = source_path.open if hasattr(source_path, "open") else open

    # 3. Extract the graphs
    with open_func("rb") as raw_file:
        with gzip.open(raw_file, "rt", encoding="utf-8") as graph_file:
            # (FIXED VARIABLE NAME HERE)
            with open(parsed_args.to_file, "w", encoding="utf-8") as out_file:
                for current_index, graph_string in enumerate(graph_file):
                    if current_index in indices_to_extract:
                        out_file.write(graph_string)
                        extracted_count += 1

                    # Optimization: Stop parsing early
                    if current_index >= max_index:
                        break

    print(f"Success! Extracted {extracted_count} graphs.")


if __name__ == "__main__":  # pragma: no cover
    main()
