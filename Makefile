.PHONY: install test lint format clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

test-cov:
	pytest tests/ --cov=neuralforge --cov-report=html

lint:
	ruff check neuralforge/ tests/
	black --check neuralforge/ tests/

format:
	black neuralforge/ tests/
	isort neuralforge/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache/ htmlcov/ coverage.xml

run-burgers:
	python experiments/burgers_fno1d.py --config configs/burgers_fno.yaml
