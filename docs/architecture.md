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
  strategy; `SemanticalFAQ` (backed by `InMemoryVectorIndex`/`LocalVectorDbIndex`) depends on the abstraction,
  not on any specific model.
- The `FAQ` adapters (`LexicalFAQ`, `SemanticalFAQ`) backed by concrete
  `KeywordIndex`/`VectorIndex` implementations are interchangeable strategies behind the `FAQ` port, selected at runtime by `FAQSearcherFactory`.

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
  `FAQSearcher`, `PromptBuilder`, and `LLMPrompter`; `SemanticalFAQ` receives an `Embedder`. Collaborators are constructor
  arguments, keeping the orchestration code backend-agnostic and testable.

### Adapter Pattern

- `SemanticalFAQ` (`classes/semantical_faq.py`) adapts a `VectorIndex` plus an
  `Embedder` to the `FAQ` contract (`open`/`search`/`close`/`build_index`/
  `count`). It is the only piece that knows the FAQ vocabulary: it encodes the
  question with the embedder, delegates the database work to the store, maps
  `SearchResult` objects back to `list[dict[str, str]]`, and owns the rebuild
  semantics of `build_index()` (open, `reset()`, insert). Every
  technology-specific import (psycopg, numpy) lives below it, in the store.
- This keeps one `FAQ` implementation per store family while allowing N vector
  store concretions, versus one `FAQ` class per (storage × engine) combination.

### Repository Pattern → replaced by VectorIndex port + FAQ adapter

- The previous `PgvectorRepository` (`classes/pgvector_repository.py`) mixed
  connection management, DDL, inserts and serialization behind an
  infrastructure vocabulary (`connect`, `db_setup`, `add_all`), which is why
  it did not behave like a repository.
- That class was removed and replaced by two pieces: `PgvectorIndex`
  (`classes/pgvector_index.py`), a `VectorIndex` concretions owning only
  database concerns, and `SemanticalFAQ`, the adapter described above.

### Null-object / lifecycle state guard

- `ManagedResource` (`classes/managed_resource.py`) centralizes the lifecycle:
  it holds a nullable resource, raises `RuntimeError(self.closed_message)` on
  use-while-closed, detaches finalizers on reconnect (the old
  `PgvectorRepository.connect` overwrote the finalizer and leaked the previous
  connection), and provides a context manager.
- Every concrete index (`InMemoryIndex`, `InMemoryVectorIndex`, `LocalDbIndex`,
  `LocalVectorDbIndex`, `PgvectorIndex`) uses it; none implement the pattern inline.

### Finalizer (resource-safety) pattern

- `PgvectorIndex` registers a `weakref.finalize` callback so a Postgres
  connection is closed even if `close()` is never called explicitly.

### Protocol-based Instruction

- `LLMInstruction` (`classes/llm_instructions.py`) is a structural `Protocol`
  (`prompt`, `message`) satisfied by `PipelineLLMInstruction` and
  `SearchAgentLLMInstruction` dataclasses, so the thread can be specialized for
  a pipeline answer vs. a search agent without inheriting from a base class.

## Design Decisions

### Configuration-driven composition

`SearcherConfig` (`classes/searcher_config.py`) uses a `Literal`-typed
dataclass (`index_type: "in_memory" | "local" | "repository"`, optional
`vector_engine: "sentence" | None`, optional `local_index_path`, optional
`database_url`). The factory
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

`LexicalFAQ` (via `LocalDbIndex`) and `SemanticalFAQ` (via `LocalVectorDbIndex`) persist indexes in a plain SQLite file via
`sqlitesearch.TextSearchIndex`/`VectorSearchIndex` (vector index in `ivf`
mode), making local RAG runs reproducible without a server.

### Data Transfer Object for search hits

- `SearchResult` (`classes/vector_index.py`) is a frozen dataclass carrying
  `document`, `distance` and `rank`. It keeps the cosine distance available at
  the store boundary without leaking pgvector row objects upward, and lets the
  adapter decide what to expose.

### build_index rebuilds the FAQ content

- Every `FAQ` implementation treats `build_index()` as a *rebuild* of the FAQ
  content fetched from `RawFaq`, not as an append. `InMemoryIndex`/`InMemoryVectorIndex`
  construct a fresh index object; `LocalDbIndex`/`LocalVectorDbIndex` call `fit()`
  again over the same database path.
- `SemanticalFAQ.build_index()` follows the same contract: it opens the store if
  needed, calls `store.reset()` (drop and recreate the `documents` table) and
  then inserts the freshly embedded FAQ items. Callers never have to connect or
  reset by hand, so `module_2/pgingestion.py` is just
  `SemanticalFAQ(PgvectorIndex(dsn, dimension=embedder.embedding_dimension), embedder).build_index()` (as composed by `FAQSearcherFactory`).
- Consequence: running the ingestion twice leaves the same document count
  (1427), and a brand new database (no `documents` table) is built from scratch
  with no prior setup step.

### pgvector index choices

- `PgvectorIndex.reset` drops and recreates the `documents` table and creates
  an HNSW index with `vector_cosine_ops`, matching the unit-normalized
  embeddings produced by MiniLM. Vectors are inserted as `[..]::vector` text
  literals to keep the dependency set minimal (no pgvector adapter).
- The table dimension comes from the required `dimension` constructor argument,
  which callers derive from the embedder (`SentenceEmbedder.embedding_dimension`),
  so the store is not silently coupled to a single embedding model.
- `search` uses the pgvector cosine operator `<=>` and selects the distance
  back, ordering by it in SQL.
- Statements run through a single `_execute` helper that translates a missing
  `documents` table into `RuntimeError("The documents table does not exist yet.
  Build the index first.")`, so a search before `build_index()` fails with the
  library's vocabulary instead of a raw `psycopg` error.

### Open-state query instead of probing

- `VectorIndex` declares an `is_open` property (implemented by the
  `ManagedResource` mixin). `SemanticalFAQ.open()` asks `store.is_open` instead of
  probing the store with `count()`, which also makes opening a store whose table
  does not exist yet perfectly valid.

### Data source access behind a lazy property

`RawFaq.items` fetches the course index and every course FAQ JSON on first
access through the property, so constructing the object is cheap and the
network fetch is deferred until an index build actually needs the data.

### Store configuration via environment

- Clients resolve the `DATABASE_URL` environment variable (with `python-dotenv`)
  and pass the DSN explicitly to `PgvectorIndex`; the store requires it as a
  constructor argument and never reads the environment itself.
