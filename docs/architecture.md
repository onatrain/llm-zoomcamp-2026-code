# Architecture of the `classes` Library

The `classes` package is a small reusable library that implements FAQ search and
Retrieval-Augmented Generation (RAG) over the DataTalks.Club FAQ corpus, plus an
agentic tool-calling stack. It is consumed by the scripts in `module_1/` and
`module_2/`.

## Design Patterns

### Strategy Pattern

- `Embedder` (`classes/embedder.py`) is an abstract base class with two
  interchangeable strategies for turning text into vectors: `embed_texts`
  (batch) and `encode_text` (single). `SentenceEmbedder` is the concrete
  strategy; `InMemoryVectorFAQ` and `LocalVectorFAQ` depend on the abstraction,
  not on any specific model.
- The `FAQ` implementations (`InMemoryFAQ`, `LocalFAQ`, `InMemoryVectorFAQ`,
  `LocalVectorFAQ`) are interchangeable strategies behind the `FAQ` abstract
  interface, selected at runtime by `FAQSearcherFactory`.

### Template Method / Uniform Interface

- `FAQ` (`classes/faq.py`) defines a uniform lifecycle of `open()`, `search()`,
  `close()`, `build_index()`, and a `count` property. Each subclass fills in the
  concrete behavior while RAG pipeline code talks only to this interface.
- `FAQ` uses Python's `ABC` plus `@abstractmethod` to enforce the contract;
  `Embedder` uses a plain `ABC` with `NotImplementedError` bodies.

### Factory Pattern

- `FAQSearcherFactory.create(config)` (`classes/faq_searcher_factory.py`) maps a
  `SearcherConfig` (index type × vector engine) to a concrete `FAQ`
  implementation and wraps it in a `FAQSearcher`. Callers never instantiate
  search backends directly.

### Builder Pattern

- `PromptBuilder` (`classes/prompt_builder.py`) accumulates a question plus
  retrieved documents into a `Thread`, returning the thread for further use.
- `Thread` itself acts as a builder for the OpenAI Responses API input: callers
  append user messages, assistant text, function calls, and tool outputs via
  small additive methods instead of constructing the list of items by hand.
- `Toolset` collects `FunctionToolParam` definitions paired with their Python
  callbacks and emits two derived views (definitions list, name→callback map).

### Dependency Injection

- `Agent` receives its `LLMPrompter` and `Thread`; `RAGPipeline` receives
  `FAQSearcher`, `PromptBuilder`, and `LLMPrompter`; `InMemoryVectorFAQ` and
  `LocalVectorFAQ` receive an `Embedder`. Collaborators are constructor
  arguments, keeping the orchestration code backend-agnostic and testable.

### Repository Pattern

- `PgvectorRepository` (`classes/pgvector_repository.py`) hides all SQL,
  connection management, and DDL (`CREATE EXTENSION`, table creation, HNSW
  index) behind `connect()`, `db_setup()`, `bulk_insert()`, and `close()`.

### Null-object / lifecycle state guard

- Every FAQ and the repository track an "opened" state with a nullable
  resource (`self._index = None` when closed) and raise
  `RuntimeError("FAQ closed. You must open it first.")` on misuse.

### Finalizer (resource-safety) pattern

- `LocalFAQ`, `LocalVectorFAQ`, and `PgvectorRepository` register a
  `weakref.finalize` callback so SQLite/Postgres connections are closed even if
  `close()` is never called explicitly; the finalizer is detached in `close()`
  to avoid a double close during GC.

### Protocol-based Instruction

- `LLMInstruction` (`classes/llm_instructions.py`) is a structural `Protocol`
  (`prompt`, `message`) satisfied by `PipelineLLMInstruction` and
  `SearchAgentLLMInstruction` dataclasses, so the thread can be specialized for
  a pipeline answer vs. a search agent without inheriting from a base class.

## Design Decisions

### Configuration-driven composition

`SearcherConfig` (`classes/searcher_config.py`) uses a `Literal`-typed
dataclass (`index_type: "in_memory" | "local"`, optional
`vector_engine: "sentence" | None`, optional `local_index_path`). The factory
validates the combination with `match` statements and raises `ValueError` for
unknown combinations — one place to evolve when a new backend is added.

### Keyword vs. vector engines behind one interface

The same `FAQ.search` signature serves minsearch `Index` (keyword fields
`question`/`section`/`answer`, keyword field `course`) and minsearch/SQLite
`VectorSearch`/`VectorSearchIndex` (semantic search over question+answer
embeddings). `boost_dict`/`filter_dict`/`num_results` are kept in the common
signature, and vector backends simply ignore `boost_dict`.

### Local embedding with forced offline mode

`SentenceEmbedder` pins `all-MiniLM-L6-v2` (384 dims, matching the
`embedding vector(384)` pgvector schema) and sets
`TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1` / `SENTENCE_TRANSFORMERS_HOME`
at construction time, so no network access or torch JIT download is attempted.

### Responses API threading over chat completions

The stack targets the OpenAI Responses API: `Thread` stores
`ResponseInputItemParam` items (messages, `function_call`, `function_call_output`),
`LLMPrompter` returns flattened text or a parsed `AgentResponse`, and the raw
client `store=False` is used so state lives in `Thread`, not server-side.

### Typed, discriminated model of model output

`LLMPrompter._process_output` converts SDK output items into a small
discriminated union — `MessageResponse` | `FunctionCallResponse` — so the
`Agent` loop can pattern-match with `match` and avoid leaking SDK types above
the prompter boundary.

### Agentic loop termination rule

`Agent.invoke` iterates until a full `AgentResponse` contains **no**
`FunctionCallResponse`; only then are the accumulated assistant message texts
joined and returned. Function calls are dispatched through the name→callback
map from `Toolset`, and each result is appended back to the thread as a tool
output before the next turn.

### Thread owns conversation state; PromptBuilder fills it

`RAGPipeline.answer` performs retrieval, then `PromptBuilder.build` renders the
context block (`section`, `Q:`, `A:` lines) into the user message via
`utils.build_prompt`, appends it to the shared `Thread`, and `LLMPrompter`
generates from the current thread including the system instruction. The
thread, not the pipeline, is the single source of conversation history.

### SQLite as the local vector/keyword store

`LocalFAQ`/`LocalVectorFAQ` persist indexes in a plain SQLite file via
`sqlitesearch.TextSearchIndex`/`VectorSearchIndex` (vector index in `ivf`
mode), making local RAG runs reproducible without a server.

### pgvector index choices

`PgvectorRepository.db_setup` drops and recreates the `documents` table and
creates an HNSW index with `vector_cosine_ops`, matching the unit-normalized
embeddings produced by MiniLM. Vectors are inserted as `[..]::vector` text
literals to keep the dependency set minimal (no pgvector adapter).

### Data source access behind a lazy property

`RawFaq.items` fetches the course index and every course FAQ JSON on first
access through the property, so constructing the object is cheap and the
network fetch is deferred until an index build actually needs the data.

### Repository configuration via environment

`PgvectorRepository` reads `DATABASE_URL` (and all LLM clients read
`LLM_API_KEY`/`LLM_BASE_URL`) from `.env` through `python-dotenv`; connection
details never live in code. `PgvectorRepository.__repr__` exposes
user@host:port/dbname without credentials.
