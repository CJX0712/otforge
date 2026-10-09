PY ?= python
PKG := otforge

.PHONY: help install lint format check test demo clean docker

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

lint:  ## static analysis
	$(PY) -m ruff check $(PKG)/ examples/ tests/

format:  ## apply the formatter
	$(PY) -m ruff format $(PKG)/ examples/ tests/

check: lint  ## lint + formatter check (what CI runs)
	$(PY) -m ruff format --check $(PKG)/ examples/ tests/

test:  ## run the pytest suite
	$(PY) -m pytest tests/ -q

demo:  ## benchmark + gate + determinism + ablation + barycenter
	$(PY) -m examples.run_demo

run:  ## CLI benchmark with the pre-registered gate
	$(PY) -m $(PKG).cli run --seeds 0,1,2

clean:  ## remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info benchmark.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker:  ## build the reproducible image
	docker build -t otforge:0.1.0 .
