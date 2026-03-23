import gzip

import pytest
from pyg6data.cli import _parse_args, main


def test_parse_args_with_order():
    args = _parse_args(["-o", "6", "-l", "0", "1", "-tf", "out.g6"])
    assert args.order == 6
    assert args.list == [0, 1]
    assert args.to_file == "out.g6"
    assert args.from_file is None


def test_parse_args_with_file():
    args = _parse_args(["-ff", "input.g6.gz", "-l", "5", "-tf", "out.g6"])
    assert args.from_file == "input.g6.gz"
    assert args.order is None
    assert args.list == [5]


def test_parse_args_rejects_both_order_and_file():
    with pytest.raises(SystemExit):
        _parse_args(["-o", "6", "-ff", "input.g6.gz", "-l", "0", "-tf", "out.g6"])


def test_main_extracts_from_internal_data(tmp_path):
    out_file = tmp_path / "output.g6"
    main(["-o", "6", "-l", "0", "1", "-tf", str(out_file)])
    lines = out_file.read_text().splitlines()
    assert len(lines) == 2


def test_main_extracts_from_custom_file(tmp_path):
    # Create a small .g6.gz file with known content
    g6_lines = b"E?`g\nE?lo\nE?qo\n"
    gz_path = tmp_path / "test.g6.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(g6_lines)

    out_file = tmp_path / "output.g6"
    main(["-ff", str(gz_path), "-l", "0", "2", "-tf", str(out_file)])
    lines = out_file.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == "E?`g"
    assert lines[1] == "E?qo"


def test_main_invalid_order(capsys):
    with pytest.raises(SystemExit):
        main(["-o", "99", "-l", "0", "-tf", "out.g6"])


def test_main_missing_file(capsys):
    with pytest.raises(SystemExit):
        main(["-ff", "/nonexistent/file.g6.gz", "-l", "0", "-tf", "out.g6"])
