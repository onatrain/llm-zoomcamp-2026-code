"""Semantical FAQ adapter for any vector index plus an embedder."""

from typing import Any

from classes.embedder import Embedder
from classes.faq import FAQ
from classes.raw_faq import RawFaq
from classes.vector_index import Vector, VectorIndex


class SemanticalFAQ(FAQ):
    """FAQ adapter that searches documents through a vector index."""

    def __init__(self, index: VectorIndex, embedder: Embedder) -> None:
        self._index = index
        self._embedder = embedder

    def open(self) -> None:
        if (
            hasattr(self._index, "dimension")
            and self._index.dimension != self._embedder.embedding_dimension
        ):
            raise ValueError("Index dimension does not match embedder dimension")

        if not self._index.is_open:
            self._index.open()

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        num_results: int = 5,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        query_vector: Vector = self._embedder.encode_text(question)

        return self._index.search(
            query_vector, filter_dict=filter_dict, num_results=num_results
        )

    def close(self) -> None:
        self._index.close()

    def build_index(self) -> None:
        """Rebuild the index from the current FAQ content.

        The previous content is only replaced once the new content has been
        fetched and embedded, so failed downloads or embeddings leave the
        index untouched.

        Raises:
            RuntimeError: If the FAQ source returns no documents.
            ValueError: If the embedder returns an unexpected amount of vectors
                or vectors with a dimension different from the declared one.
        """
        faq_items = RawFaq().items
        if not faq_items:
            raise RuntimeError(
                "The FAQ source returned no documents; the index was not modified"
            )

        texts = [" ".join((item["question"], item["answer"])) for item in faq_items]

        embeddings = self._embedder.embed_texts(texts)

        if len(embeddings) != len(faq_items):
            raise ValueError(
                f"Got {len(faq_items)} documents and {len(embeddings)} embeddings, "
                "they must be the same amount"
            )

        if embeddings.shape[1] != self._embedder.embedding_dimension:
            raise ValueError(
                "Embedding dimension does not match the embedder dimension"
            )

        self.open()

        self._index.reset()

        self._index.load(faq_items, list(embeddings))

    @property
    def count(self) -> int:
        return self._index.size()
