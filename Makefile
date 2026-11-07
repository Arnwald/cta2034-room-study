PYTHON ?= .venv/bin/python

.PHONY: all test study figures paper clean

all: study figures paper

test:
	$(PYTHON) -m pytest -q

study:
	$(PYTHON) scripts/run_study.py

figures:
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/make_figures.py fr

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex main_fr.tex

clean:
	cd paper && latexmk -C main.tex main_fr.tex
