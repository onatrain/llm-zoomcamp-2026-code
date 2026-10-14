# AGENTS.md

## Project

Course code for LLM Zoomcamp 2026: RAG pipelines and agents over FAQ data. Python 3.12, managed with uv.

## Layout

- `classes/` — reusable library (FAQ searchers, embedders, RAG pipeline, Agent/Toolset stack, pgvector repo). Entrypoint for config composition: `classes/faq_searcher_factory.py`.
- `module_1/` — keyword-search RAG scripts (minsearch, sqlitesearch, function calling, agentic loop).
- `module_2/` — vector-search scripts (sentence-transformers, numpy dot-product search, pgvector).
- `utils/` — shared helpers (`llm`, `build_prompt`), importable from the repo root.
- `control/` — scratch/experimental files; NOT part of the runnable package (gitignored).

## Running scripts

- Install deps: `uv sync`
- Some scripts in `module_1`/`module_2` import each other with bare sibling imports (e.g. `from L5_minsearch import ...`), so run them with `PYTHONPATH` pointing at the repo root:
  `PYTHONPATH=$PWD uv run python module_1/L6_build_prompt.py`
  (You can use `uv run python -m module_1.L9_rag_sqlitesearch` for self-contained scripts.)
- Cross-module imports of `classes`/`utils` require repo root on `PYTHONPATH` or running from the module dir.

## Environment

- Secrets file `.env` exists at project root directory but it is not accesible; it contains `LLM_API_KEY`, `LLM_BASE_URL`, `DATABASE_URL` value definitions.
- sentence-transformers runs offline (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`) with models cache dir `./models`; do not remove those env vars in scripts.
- pgvector scripts need the docker service: `docker compose up -d db` (env from `.env.docker`, port 5432). Data persists in `pgvector_data/`.

## Checks

- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Import sorting: `uv run isort .`
- Types: `uv run mypy .` (strict mode; `ignore_missing_imports = true`)
- No test suite exists; verify by running the relevant script.

## Conventions

- All comments, user messages, and docstrings in English; docstrings use Google style (Args/Returns), no usage examples.
- Line length 88 (ruff/isort/mypy config in `pyproject.toml`).
- torch is pinned to the CPU index in `pyproject.toml` (`pytorch-cpu`) — keep that source.
