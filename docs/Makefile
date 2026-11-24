# Minimal makefile for the DIMSWE Sphinx documentation.
#
# Usage (from the repository root):
#   make -C docs html          build HTML into docs/_build/html
#   make -C docs strict        build HTML, treating warnings as errors (what CI runs)
#   make -C docs linkcheck     check external links
#   make -C docs clean         remove docs/_build
#
# Set DIMSWE_DOCS_MOCK=1 to mock the Firedrake stack (see docs/sphinx/conf.py).

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = sphinx
BUILDDIR      = _build

.PHONY: help strict Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# HTML build with warnings as errors; --keep-going reports all warnings, not just the first.
strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going $(SPHINXOPTS) $(O)

# Route every other target (html, linkcheck, clean, ...) to sphinx-build's make mode.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
