"""Sphinx configuration for the Pycliques monorepo documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC_DIRS = sorted(
    (path / "src" for path in PROJECT_ROOT.joinpath("packages").iterdir() if (path / "src").exists()),
    key=lambda p: p.parent.name,
)
for src_dir in PACKAGE_SRC_DIRS:
    sys.path.insert(0, str(src_dir))

PACKAGE_NAMES = [src_dir.parent.name for src_dir in PACKAGE_SRC_DIRS]

project = "Pycliques Graph Theory Ecosystem"
author = "Rafael Villarroel"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options: dict[str, object] = {
    "logo": {
        "text": project,
    },
    "show_toc_level": 2,
}

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

doctest_global_setup = "\n".join([f"import {name}" for name in PACKAGE_NAMES])

todo_include_todos = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
language = "en"

latex_engine = "pdflatex"
latex_use_xindy = True
latex_documents = [
    ("index", "pycliques.tex", project, author, "manual"),
]
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
\usepackage{enumitem}
\setlist{nosep}
""",
}
