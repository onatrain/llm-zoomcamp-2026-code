"""Registry/provider based factory for FAQ searchers."""

from typing import Callable

from classes.faq import FAQ
from classes.faq_searcher import FAQSearcher
from classes.searcher_config import SearcherConfig

_PROVIDER_REGISTRY: dict[
    tuple[str, str | None],
    Callable[[SearcherConfig, object | None], FAQ],
] = {}


def _in_memory_lexical(config: SearcherConfig, embedder: object | None = None) -> FAQ:
    from classes.in_memory_index import InMemoryIndex
    from classes.lexical_faq import LexicalFAQ

    return LexicalFAQ(InMemoryIndex())


def _in_memory_semantic(config: SearcherConfig, embedder: object | None = None) -> FAQ:
    from classes.embedder import Embedder
    from classes.in_memory_vector_index import InMemoryVectorIndex
    from classes.semantical_faq import SemanticalFAQ
    from classes.sentence_embedder import SentenceEmbedder

    resolved_embedder: Embedder = (
        embedder if isinstance(embedder, Embedder) else SentenceEmbedder()
    )
    return SemanticalFAQ(InMemoryVectorIndex(), resolved_embedder)


def _local_lexical(config: SearcherConfig, embedder: object | None = None) -> FAQ:
    from classes.lexical_faq import LexicalFAQ
    from classes.local_db_index import LocalDbIndex

    path = config.local_index_path
    if path is None or not path.strip():
        raise ValueError("Local DB path must be provided")

    return LexicalFAQ(LocalDbIndex(path))


def _local_semantic(config: SearcherConfig, embedder: object | None = None) -> FAQ:
    from classes.local_vector_db_index import LocalVectorDbIndex
    from classes.semantical_faq import SemanticalFAQ
    from classes.sentence_embedder import SentenceEmbedder

    path = config.local_index_path
    if path is None or not path.strip():
        raise ValueError("Local DB path must be provided")

    from classes.embedder import Embedder

    resolved_embedder: Embedder = (
        embedder if isinstance(embedder, Embedder) else SentenceEmbedder()
    )
    return SemanticalFAQ(LocalVectorDbIndex(path), resolved_embedder)


def _repository_semantic(config: SearcherConfig, embedder: object | None = None) -> FAQ:
    from classes.pgvector_index import PgvectorIndex
    from classes.semantical_faq import SemanticalFAQ
    from classes.sentence_embedder import SentenceEmbedder

    dsn = config.database_url
    if dsn is None or not dsn.strip():
        raise ValueError("Database URL must be provided")

    from classes.embedder import Embedder

    resolved_embedder: Embedder = (
        embedder if isinstance(embedder, Embedder) else SentenceEmbedder()
    )
    return SemanticalFAQ(
        PgvectorIndex(dsn, dimension=resolved_embedder.embedding_dimension),
        resolved_embedder,
    )


_PROVIDER_REGISTRY[("in_memory", None)] = _in_memory_lexical
_PROVIDER_REGISTRY[("in_memory", "sentence")] = _in_memory_semantic
_PROVIDER_REGISTRY[("local", None)] = _local_lexical
_PROVIDER_REGISTRY[("local", "sentence")] = _local_semantic
_PROVIDER_REGISTRY[("repository", "sentence")] = _repository_semantic


class FAQSearcherFactory:
    def __init__(self, embedder: object | None = None) -> None:
        self._embedder = embedder

    def create(
        self, config: SearcherConfig, embedder: object | None = None
    ) -> FAQSearcher:
        resolved_embedder = (
            embedder if embedder is not None else self._embedder
        )

        provider = _PROVIDER_REGISTRY.get(
            (config.index_type, config.vector_engine)
        )
        if provider is None:
            raise ValueError(
                f"Unknown index type/vector engine: "
                f"{config.index_type}/{config.vector_engine}"
            )

        index = provider(config, resolved_embedder)

        # In-memory backends materialise their data during `open()` in the
        # original implementations; keep that convenience for callers.
        if config.index_type == "in_memory":
            index.build_index()
        else:
            index.open()

        return FAQSearcher(index)
