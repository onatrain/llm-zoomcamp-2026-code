## LLM Zoomcamp 2026 Course Code

This is my participation in the free course LLM Zoomcamp started on June 2026. I publish code exercises from
the course here, made my way, because Python scripts are more friendly than Jupyter Notebooks for me.

In this repo you will find tests, prototypes and course code following the course contents published at
[LLM Zoomcamp repo](https://github.com/DataTalksClub/llm-zoomcamp/tree/main)

Many thanks to [Alexei Grigoriev](https://github.com/alexeygrigorev) for running Datatalks Club and bringing us these 
really good Zoomcamps.

## Using this code

Clone this repo
```bash
git clone https://github.com/onatrain/llm-zoomcamp-2026-code.git your-local-directory/
```

Install dependencies
```bash
cd your-local-directory/
uv sync
```

Create your own .env file for your secrets
```bash
cp .env.example .env
```

Edit your .env file and set your llm provider api key, base_url and postgres/pgvector dns into LLM_API_KEY, LLM_BASE_URL and DATABASE_URL 

Create your own .env.docker file for your docker containerized services secrets
```bash
cp .env.docker.example .env.docker
```

Edit your .env.docker file and set your db name, db user and db password into the service variables 

Run some program. Any program, specially those that import other programs, can be run with PYTHONPATH. By example:
```bash
PYTHONPATH=$PWD uv run python module_1/L6_build_prompt.py
```

Self-contained programs can be run as a module. By example:
```bash
uv run python -m module_1.L9_rag_sqlitesearch
```

## Contents at a glance

### classes

A small library of reusable components for building RAG pipelines and agents over the course FAQ data. Features include:

- Abstract `FAQ` interface with in-memory, local SQLite, and pgvector implementations
- Ports/Indexes: `VectorIndex` and `KeywordIndex`; adapters: `SemanticalFAQ` and `LexicalFAQ`
- Keyword search (minsearch & sqlitesearch) and vector search (sentence-transformers embeddings)
- `FAQSearcherFactory` (provider registry based) and `SearcherConfig` to compose searchers from a single config
- `ManagedResource` mixin for the open/close lifecycle, finalizers, and context managers of external resources
- Search parameters passed through `extra_params` (OpenAI-style), including lexical-only `boost_dict`
- Prompt building (`PromptBuilder`, `Thread`, `LLMInstruction`) and RAG orchestration (`RAGPipeline`)
- An `Agent` / `LLMPrompter` / `Toolset` stack for tool-calling agentic loops
- Data ingestion helpers (`RawFaq`, `SentenceEmbedder`, `PgvectorIndex`)

### module_1

Scripts demonstrating the basics of LLM-powered search over the course FAQ. Features include:

- Connecting to an OpenAI-compatible LLM endpoint and listing models
- Plain LLM question answering and prompting with injected context
- A deliberately naive exact-match "RAG" baseline
- Fetching and flattening the FAQ JSON sources
- In-memory keyword search with minsearch, field boosting, and course filtering
- Building prompts from retrieved context
- Full RAG pipelines backed by in-memory and local SQLite search
- Function calling with the Responses API and a multi-turn agentic loop

### module_2

Scripts extending the RAG pipeline with vector search and embeddings. Features include:

- Generating sentence embeddings with `all-MiniLM-L6-v2` via sentence-transformers
- Batch embedding the FAQ dataset into numpy vectors
- Brute-force vector similarity search with dot products and top-k ranking
- Approximate vector search with minsearch `VectorSearch` and course filters
- Building local and in-memory vector FAQ indexes
- RAG pipelines using sentence-embedding vector search, backed by local SQLite and pgvector

## Class relationship

- FAQ, VectorIndex, KeywordIndex, and Embedder are abstract classes.
- ManagedResource is the generic mixin base of every concrete index (InMemoryIndex, InMemoryVectorIndex, LocalDbIndex, LocalVectorDbIndex, PgvectorIndex).
- SemanticalFAQ implements FAQ and adapts any VectorIndex using an Embedder.
- LexicalFAQ implements FAQ and adapts any KeywordIndex.
- PgvectorIndex implements VectorIndex (and ManagedResource) and uses PostgreSQL/pgvector.
- LocalVectorDbIndex implements VectorIndex (and ManagedResource) and uses sqlitesearch.
- InMemoryVectorIndex implements VectorIndex (and ManagedResource) and uses minsearch.
- InMemoryIndex implements KeywordIndex (and ManagedResource) and uses minsearch.
- LocalDbIndex implements KeywordIndex (and ManagedResource) and uses sqlitesearch.
- SentenceEmbedder implements Embedder and uses sentence-transformers.
- RawFaq supplies the FAQ items consumed by SemanticalFAQ and LexicalFAQ in build_index().
- FAQSearcher wraps a FAQ instance.
- FAQSearcherFactory creates a FAQSearcher from a SearcherConfig.
- SearcherConfig selects the index type and vector engine.
- RAGPipeline composes a FAQSearcher, a PromptBuilder, and an LLMPrompter.

## License

All code published here is [MIT](https://mit-license.org/) licensed.
