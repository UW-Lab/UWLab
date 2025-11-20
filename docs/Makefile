# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: multi-docs
multi-docs:
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@cp _redirect/index.html $(BUILDDIR)/index.html

.PHONY: current-docs
current-docs:
	@rm -rf "$(BUILDDIR)/current"
	@$(SPHINXBUILD) -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS)

.PHONY: serve-current
serve-current: current-docs
	@echo "Serving docs at http://127.0.0.1:8000 (Ctrl+C to stop)"
	@python -m http.server 8000 --bind 127.0.0.1
