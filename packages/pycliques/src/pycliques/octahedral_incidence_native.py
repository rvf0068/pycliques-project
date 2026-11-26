"""Explicitly built native adaptive cut walk for implicit intersection graphs."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

MAX_WALK_ORDER = 1280
MAX_WALK_GROUND = 64


def _source() -> Path:
    return Path(__file__).with_name("_octahedral_incidence_walk.cpp")


def _cached(cache_dir: Path | None = None) -> Path:
    tag = hashlib.sha256(_source().read_bytes()).hexdigest()[:16]
    root = (
        cache_dir
        if cache_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "pycliques"
    )
    return root / f"incidence-walk-{tag}"


def build_incidence_native(
    *, compiler: str | None = None, cache_dir: Path | None = None
) -> Path:
    """Explicitly compile the optional GNU C++17/libstdc++ walk."""
    cxx = compiler or os.environ.get("CXX", "g++")
    if shutil.which(cxx) is None:
        raise RuntimeError(f"C++ compiler {cxx!r} not found")
    destination = _cached(cache_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        exe = Path(directory) / "walk"
        run = subprocess.run(
            [cxx, "-O3", "-std=c++17", str(_source()), "-o", str(exe)],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode:
            raise RuntimeError(f"Incidence walk build failed: {run.stderr[-2000:]}")
        exe.replace(destination)
    return destination


def find_incidence_native() -> Path:
    """Locate the version-matched executable; never compile implicitly."""
    path = _cached()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            "Build the incidence walk with "
            "python -m pycliques.octahedral_incidence_native --build"
        )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build optional incidence cut walk")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("pass --build to compile the optional backend")
    print(build_incidence_native())
