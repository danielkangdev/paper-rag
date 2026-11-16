API ?= http://localhost:8000
PY ?= python

.PHONY: demo up down install test lint ingest ablation answer-eval readme

# docker compose up -d, then: ingest 5 papers through the API and ask one question.
# (Without make: run the two commands below by hand.)
demo:
	docker compose up -d --build
	$(PY) scripts/demo.py --api $(API)

up:
	docker compose up -d --build

down:
	docker compose down

install:
	$(PY) -m pip install -e ".[models,dev]"

lint:
	ruff check scirag tests scripts

test:
	pytest -q

ingest:
	scirag ingest --file data/corpus.txt

ablation:
	$(PY) -m scirag.eval.run_ablation

answer-eval:
	$(PY) -m scirag.eval.run_answer_eval

readme:
	$(PY) -m scirag.eval.report --readme
