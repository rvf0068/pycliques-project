# Minimal makefile for Sphinx documentation

SPHINXBUILD ?= sphinx-build
SPHINXPROJ = PycliquesDocs
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help clean html doctest latexpdf pdf

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf "$(BUILDDIR)"

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

doctest:
	$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"

latexpdf:
	$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)"

pdf: latexpdf
