# AGENTS.md

## Project

Course code for LLM Zoomcamp 2026: RAG pipelines and agents over FAQ data. Python 3.12, managed with uv.

## Layout

- `classes/` — reusable library (FAQ searchers, embedders, RAG pipeline, Agent/Toolset stack, vector stores). Entrypoint for config composition: `classes/faq_searcher_factory.py`.
  - Ports and contracts:
    - `faq.py` — `FAQ` contract (`open`/`search`/`close`/`build_index`/`count`).
    - `vector_index.py` — `VectorIndex` port for vector indexes.
    - `keyword_index.py` — `KeywordIndex` port for lexical indexes.
    - `embedder.py` — `Embedder` port for text embedding.
    - `managed_resource.py` — `ManagedResource` mixin: open/close lifecycle, finalizers, context manager.
  - FAQ adapters:
    - `semantical_faq.py` — `SemanticalFAQ`, `FAQ` adapter over any `VectorIndex` + `Embedder`.
    - `lexical_faq.py` — `LexicalFAQ`, `FAQ` adapter over any `KeywordIndex`.
  - Concrete indexes:
    - `in_memory_index.py` — `InMemoryIndex` (`KeywordIndex` over minsearch).
    - `in_memory_vector_index.py` — `InMemoryVectorIndex` (`VectorIndex` over minsearch).
    - `local_db_index.py` — `LocalDbIndex` (`KeywordIndex` over sqlitesearch).
    - `local_vector_db_index.py` — `LocalVectorDbIndex` (`VectorIndex` over sqlitesearch).
    - `pgvector_index.py` — `PgvectorIndex` (`VectorIndex` over PostgreSQL/pgvector).
  - Composition and orchestration:
    - `faq_searcher.py` — `FAQSearcher` wraps a `FAQ`.
    - `faq_searcher_factory.py` — `FAQSearcherFactory` and provider registry.
    - `searcher_config.py` — `SearcherConfig` dataclass.
    - `rag_pipeline.py` — `RAGPipeline` composes `FAQSearcher`, `PromptBuilder`, and `LLMPrompter`.
  - Agent stack:
    - `agent.py` — `Agent` agentic loop.
    - `llm_prompter.py` — `LLMPrompter` and response models.
    - `toolset.py` — `Toolset` definitions and callbacks.
    - `thread.py` — `Thread` conversation state.
    - `llm_instructions.py` — `LLMInstruction` protocol and implementations.
    - `prompt_builder.py` — `PromptBuilder`.
  - Data and embeddings:
    - `raw_faq.py` — `RawFaq` FAQ corpus fetcher.
    - `sentence_embedder.py` — `SentenceEmbedder` (offline sentence-transformers).
- `module_1/` — keyword-search RAG scripts (minsearch, sqlitesearch, function calling, agentic loop).
- `module_2/` — vector-search scripts (sentence-transformers, numpy dot-product search, pgvector).
- `utils/` — shared helpers (`llm`, `build_prompt`, `normalize`), importable from the repo root.
- `control/` — scratch/experimental files; NOT part of the runnable package (gitignored).

## Documentation & Specs

- Consult `README.md` for the project overview, setup, contents at a glance, and class relationships.
- Consult `docs/architecture.md` for design patterns, design decisions, and component specifications.
- Read these files before modifying `classes/` when a description or specification of a component is needed.

## Running scripts

- Install deps: `uv sync`
- Some scripts in `module_1`/`module_2` import each other with bare sibling imports (e.g. `from L5_minsearch import ...`), so run them with `PYTHONPATH` pointing at the repo root:
  `PYTHONPATH=$PWD uv run python module_1/L6_build_prompt.py`
  (You can use `uv run python -m module_1.L9_rag_sqlitesearch` for self-contained scripts.)
- Cross-module imports of `classes`/`utils` require repo root on `PYTHONPATH` or running from the module dir.

## Environment

- Secrets file `.env` exists at project root directory but it is not accessible; it contains `LLM_API_KEY`, `LLM_BASE_URL`, `DATABASE_URL` value definitions.
- sentence-transformers must run offline: `classes/sentence_embedder.py` sets `TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1` at import time (huggingface_hub reads those flags only once, when imported) and resolves the cached model directory instead of the model id. `local_model_path()` searches `SENTENCE_TRANSFORMERS_HOME` and the HF Hub cache (`HF_HUB_CACHE`/`HF_HOME`, default `~/.cache/huggingface/hub`). Loading a local directory skips Hub resolution entirely, so no weights download is ever attempted. Use `load_transformer()` instead of building `SentenceTransformer` in scripts; do not remove the offline env vars.
- pgvector scripts need the docker service: `docker compose up -d db` (env from `.env.docker`, port 5432). Data persists in `pgvector_data/`.
- `module_2/pgingestion.py` is the canonical pgvector ingestion: it builds `SemanticalFAQ(PgvectorIndex(dsn, dimension=embedder.embedding_dimension), embedder)` and calls `build_index()`, which opens the store, resets it and inserts the embedded FAQ (no manual `connect()`/`db_reset()`). It is safe to run repeatedly; a fresh database needs no prior setup.

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

- None.
