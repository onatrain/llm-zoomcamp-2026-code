# AGENTS.md

## Project

Course code for LLM Zoomcamp 2026: RAG pipelines and agents over FAQ data. Python 3.12, managed with uv.

## Layout

- `classes/` — reusable library (FAQ searchers, embedders, RAG pipeline, Agent/Toolset stack, vector stores). Entrypoint for config composition: `classes/faq_searcher_factory.py`.
  - `faq.py` — `FAQ` contract (open/search/close/build_index/count); other indexes are `FAQ` adapters. `build_index()` always *rebuilds* the FAQ content fetched from `RawFaq`; for pgvector that means the adapter resets the store itself.
  - `vector_index.py` — `VectorIndex` port + `SearchResult` for vector indexes.
- `keyword_index.py` — `KeywordIndex` port for lexical indexes.
  - `managed_resource.py` — `ManagedResource` mixin: open/close lifecycle, finalizers, context manager.
  - `pgvector_index.py` — `PgvectorIndex`, the psycopg/pgvector `VectorIndex` implementation.
  - `semantical_faq.py` — `SemanticalFAQ`, `FAQ` adapter over any `VectorIndex` + `Embedder`.
- `lexical_faq.py` — `LexicalFAQ`, `FAQ` adapter over any `KeywordIndex`.
- `in_memory_index.py`, `in_memory_vector_index.py`, `local_db_index.py`, `local_vector_db_index.py` — concrete index implementations.
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
- sentence-transformers must run offline: `classes/sentence_embedder.py` sets `TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1` at import time (huggingface_hub reads those flags only once, when imported) and resolves the cached model directory instead of the model id. `local_model_path()` searches `SENTENCE_TRANSFORMERS_HOME` and the HF Hub cache (`HF_HUB_CACHE`/`HF_HOME`, default `~/.cache/huggingface/hub`). Loading a local directory skips Hub resolution entirely, so no weights download is ever attempted. Use `load_transformer()` instead of building `SentenceTransformer` in scripts; do not remove the offline env vars.
- pgvector scripts need the docker service: `docker compose up -d db` (env from `.env.docker`, port 5432). Data persists in `pgvector_data/`.
- `module_2/pgingestion.py` is the canonical pgvector ingestion: it only builds a `PgvectorFAQ` and calls `build_index()`, which opens the store, resets it and inserts the embedded FAQ (no manual `connect()`/`db_reset()`). It is safe to run repeatedly; a fresh database needs no prior setup.

## Checks

- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Import sorting: `uv run isort .`
- Types: `uv run mypy .` (strict mode; `ignore_missing_imports = true`)
- Tests: `uv run pytest`. `tests/unit` is offline and fast; `tests/integration` downloads the real FAQ corpus once per session, embeds it with the cached sentence-transformers model and exercises the non-in-memory adapters with real data. pgvector integration tests need `docker compose up -d db` and skip when `DATABASE_URL` is missing or PostgreSQL is unreachable.

## Conventions

- All comments, user messages, and docstrings in English; docstrings use Google style (Args/Returns), no usage examples.
- Line length 88 (ruff/isort/mypy config in `pyproject.toml`).
- torch is pinned to the CPU index in `pyproject.toml` (`pytorch-cpu`) — keep that source.

## Pending

- Integrate PgVectorIndex to the RAG pipeline and test it.
