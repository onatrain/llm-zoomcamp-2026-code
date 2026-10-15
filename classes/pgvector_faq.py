from classes.embedder import Embedder
from classes.faq import FAQ
from classes.raw_faq import RawFaq
from classes.vector_store import Vector, VectorStore


class PgvectorFAQ(FAQ):
    """FAQ adapter that searches documents through a vector store.

    Owns the mapping to the ``FAQ`` contract: it delegates all database work to
    the injected ``VectorStore`` and all text-to-vector work to the injected
    ``Embedder``.
    """

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        """Initialize the adapter.

        Args:
            store: The vector store holding the documents and their embeddings.
            embedder: The embedder used to encode questions into query vectors.
        """
        self._store = store
        self._embedder = embedder

    def open(self) -> None:
        """Open the vector store if it is closed.

        Raises:
            ValueError: If the store dimension and the embedder dimension differ.
        """
        if self._store.dimension != self._embedder.embedding_dimension:
            raise ValueError(
                "Store dimension "
                f"{self._store.dimension} does not match embedder dimension "
                f"{self._embedder.embedding_dimension}"
            )

        if not self._store.is_open:
            self._store.connect()

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[dict[str, str]]:
        """Search the FAQ by vector similarity.

        ``boost_dict`` is ignored: field boosting is a keyword-search feature
        that does not apply to vector similarity.

        Args:
            question: The question to search for.
            filter_dict: Optional mapping used to restrict the search. Only the
                ``course`` key is used.
            boost_dict: Ignored, kept for signature parity with other ``FAQ``
                implementations.
            num_results: Maximum number of results to return.

        Returns:
            The matching FAQ entries as dictionaries.
        """
        query_vector: Vector = self._embedder.encode_text(question)

        course = filter_dict["course"] if filter_dict else None
        results = self._store.search(
            query_vector, course=course, num_results=num_results
        )

        return [result.document for result in results]

    def close(self) -> None:
        """Close the vector store."""
        self._store.close()

    def build_index(self) -> None:
        """Rebuild the FAQ content stored in the vector store.

        Opens the store if needed, resets the stored documents and inserts the
        FAQ items fetched from the raw FAQ source, embedded with the injected
        embedder. As with every ``FAQ`` implementation, the index is
        reconstructed from scratch, so this operation is destructive: any
        previously stored FAQ content is deleted.
        """
        self.open()

        self._store.db_reset()

        faq_items = RawFaq().items

        texts = [" ".join((item["question"], item["answer"])) for item in faq_items]

        embeddings = self._embedder.embed_texts(texts)

        self._store.bulk_insert(faq_items, list(embeddings))

    @property
    def count(self) -> int:
        """Return the number of documents in the store.

        Returns:
            The document count.
        """
        return self._store.count()
