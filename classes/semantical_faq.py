"""Semantical FAQ adapter for any vector index plus an embedder."""

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
        if hasattr(self._index, "dimension") and self._index.dimension != self._embedder.embedding_dimension:
            raise ValueError(
                "Index dimension does not match embedder dimension"
            )

        if not self._index.is_open:
            self._index.open()

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[dict[str, str]]:
        query_vector: Vector = self._embedder.encode_text(question)

        course = filter_dict["course"] if filter_dict else None
        results = self._index.search(
            query_vector, course=course, num_results=num_results
        )

        return [result.document for result in results]

    def close(self) -> None:
        self._index.close()

    def build_index(self) -> None:
        self.open()

        self._index.reset()

        faq_items = RawFaq().items

        texts = [" ".join((item["question"], item["answer"])) for item in faq_items]

        embeddings = self._embedder.embed_texts(texts)

        self._index.add_all(faq_items, list(embeddings))

    @property
    def count(self) -> int:
        return self._index.size()
