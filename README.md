## LLM Zoomcamp 2026 Course Code

This is my participation in the free course LLM Zoomcamp started on June 8th, 2026. I'll publish code exercises from
the course here, made my way, because Python scripts are more friendly to me than Jupyter Notebooks.

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
uv sync
```

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
- Keyword search (minsearch) and vector search (sentence-transformers embeddings)
- `FAQSearcherFactory` and `SearcherConfig` to compose searchers from a single config
- Prompt building (`PromptBuilder`, `Thread`, `LLMInstruction`) and RAG orchestration (`RAGPipeline`)
- An `Agent` / `LLMPrompter` / `Toolset` stack for tool-calling agentic loops
- Data ingestion helpers (`RawFaq`, `SentenceEmbedder`, `PgvectorRepository`)

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

## License

All code published here is [MIT](https://mit-license.org/) licensed.