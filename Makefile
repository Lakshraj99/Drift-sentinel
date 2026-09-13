.PHONY: quick reproduce test figures

quick:
	python scripts/run_pipeline.py reproduce --quick

reproduce:
	python scripts/run_pipeline.py reproduce

test:
	pytest -q

figures:
	python -m driftsentinel evaluate
