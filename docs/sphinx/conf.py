"""Sphinx configuration for the DIMSWE documentation.

Build from the repository root with ``make -C docs html`` (output in ``docs/_build/html``) or
directly with ``sphinx-build -b html docs/sphinx docs/_build/html``.

Firedrake and its stack (UFL, FIAT, FInAT, PETSc, MPI, ...) cannot be pip-installed, so autodoc
can run in two modes:

* **Real imports** (used in CI, inside the Firedrake container): the documented modules are
  imported against the actual Firedrake installation.
* **Mocked imports**: the Firedrake stack is replaced by ``autodoc_mock_imports`` so the docs can
  be built in a plain Python environment. ``pip install .[docs]`` does not work there, because
  ``firedrake`` is a runtime dependency of the package; instead install the documentation tools
  and the pip-installable runtime dependencies directly, without installing ``dimswe`` itself
  (``conf.py`` puts the repository root on ``sys.path``)::

      pip install sphinx furo myst-parser sphinx-copybutton numpy scipy matplotlib pyyaml

  Inside a Firedrake environment, ``pip install -e .[docs]`` works as usual.

The mode is chosen by the ``DIMSWE_DOCS_MOCK`` environment variable: ``1`` forces mocking, ``0``
forces real imports, and when it is unset, mocking is used only if ``firedrake`` cannot be
imported.
"""

import importlib.util
import os
import sys
import tomllib
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------

# docs/sphinx/conf.py -> repository root. The root holds the ``dimswe`` and ``workflows``
# packages and the ``run_model`` script that some workflows import, so it goes on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _read_project_version(pyproject_path):
    """Return the version string declared in ``pyproject.toml``.

    Args:
        pyproject_path: Path to the repository's ``pyproject.toml``.

    Returns:
        The ``[project].version`` entry, or ``"unknown"`` if the file or entry is missing, so a
        packaging problem never breaks the documentation build.
    """
    try:
        with open(pyproject_path, "rb") as file:
            return tomllib.load(file)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def _use_mocked_imports():
    """Decide whether autodoc should mock the Firedrake stack.

    Returns:
        True if ``DIMSWE_DOCS_MOCK`` is ``1``/``true``/``yes``, False if it is
        ``0``/``false``/``no``, and otherwise True exactly when ``firedrake`` is not importable.
    """
    flag = os.environ.get("DIMSWE_DOCS_MOCK", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    # find_spec only locates the package; it does not import (and initialize) Firedrake.
    return importlib.util.find_spec("firedrake") is None


# ---------------------------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------------------------

project = "DIMSWE"
author = "DIMSWE developers"
copyright = "DIMSWE developers"
release = _read_project_version(REPO_ROOT / "pyproject.toml")
version = release

# ---------------------------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",       # API pages from docstrings
    "sphinx.ext.autosummary",   # per-module summary tables and stub pages
    "sphinx.ext.napoleon",      # Google-style ``Args:`` / ``Returns:`` docstrings
    "sphinx.ext.mathjax",       # LaTeX math in docstrings and pages
    "sphinx.ext.viewcode",      # links from API entries to highlighted source
    "sphinx.ext.intersphinx",   # cross-references into external projects
    "myst_parser",              # Markdown sources (existing docs/*.md, READMEs)
    "sphinx_copybutton",        # copy button on code blocks
]

# Markdown and reStructuredText are both accepted as sources.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"

# _templates holds Jinja templates (with .rst suffixes), not pages. Planning notes are
# deliberately not published (see TODO-documentation.md, "Decisions"), so they are excluded even
# if they are ever copied or symlinked into the source tree.
exclude_patterns = [
    "_build",
    "_templates",
    "**/TODO-*.md",
    "**/HANDOFF-*.md",
]

# ---------------------------------------------------------------------------------------------
# autodoc / autosummary / napoleon
# ---------------------------------------------------------------------------------------------

# Write the per-module stub pages listed in api/index.rst to api/generated/ at build time. They
# are regenerated on every build and are not under version control (see .gitignore).
autosummary_generate = True
# Document only objects defined in each module, not names pulled in by ``from .x import *``
# (``dimswe/__init__.py`` re-exports almost every module, which would duplicate every entry).
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

if _use_mocked_imports():
    # Every module that cannot be installed from PyPI (or is heavy and only needed at runtime).
    autodoc_mock_imports = [
        "firedrake",
        "ufl",
        "FIAT",
        "finat",
        "pyop2",
        "tsfc",
        "petsc4py",
        "slepc4py",
        "mpi4py",
        "h5py",
        "pyvista",
    ]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_attr_annotations = True

# ---------------------------------------------------------------------------------------------
# MyST (Markdown)
# ---------------------------------------------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",   # $...$ and $$...$$ math
    "amsmath",      # \begin{align} ... environments
    "colon_fence",  # ::: directive fences
    "deflist",
]
# Generate anchors for headings up to level 3, so the ``#section`` links used throughout
# workflows/README.md resolve when it is included.
myst_heading_anchors = 3

# ---------------------------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------------------------

# The macros of docs/hodge_laplacian.tex, so converted theory pages and docstrings can use the
# same notation. Entries with arguments are given as [definition, number_of_arguments].
mathjax3_config = {
    "tex": {
        "macros": {
            "dd": r"\mathrm{d}",
            "cod": r"\delta",
            "tr": r"\operatorname{tr}",
            "hodge": r"\star",
            "Hspace": r"\mathfrak{h}",
            "R": r"\mathbb{R}",
            "curltwo": r"\operatorname{curl_{2D}}",
            "skewgrad": r"\operatorname{skewgrad}",
            "grad": r"\operatorname{grad}",
            "curl": r"\operatorname{curl}",
            "div": r"\operatorname{div}",
            "code": [r"\texttt{#1}", 1],
            "ip": [r"\left(#1,\,#2\right)", 2],
        },
    },
}

# ---------------------------------------------------------------------------------------------
# Cross-project links
# ---------------------------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "firedrake": ("https://www.firedrakeproject.org/", None),
    "ufl": ("https://docs.fenicsproject.org/ufl/main/", None),
    "petsc4py": ("https://petsc.org/release/petsc4py/", None),
}

# ---------------------------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------------------------

html_theme = "furo"
html_title = f"DIMSWE {release}"
# No static files yet; listing a missing directory here would produce a warning, which fails
# ``-W`` builds.
html_static_path = []
# _templates/autosummary/module.rst lays out the per-module API pages (see api/index.rst).
templates_path = ["_templates"]

# Skip the prompt characters when copying shell and Python console examples.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
