"""In-memory vector index backed by ``minsearch.VectorSearch``."""

from typing import Sequence

from minsearch import VectorSearch

from classes.managed_resource import ManagedResource
from classes.vector_index import SearchResult, Vector, VectorIndex
from utils.normalize import normalize_index_param


class InMemoryVectorIndex(ManagedResource[VectorSearch], VectorIndex):
    """In-memory vector index using cosine similarity via minsearch.

    The index is volatile: its state is never persisted, so opening it always
    yields a fresh, empty instance. ``reset()`` is therefore a no-op that
    exists only to satisfy the ``VectorIndex`` contract.
    """

    closed_message = "Index closed. You must open() it first."

    def __init__(self) -> None:
        """Initialize a closed, empty in-memory vector index."""
        super().__init__()
        self._index: VectorSearch | None = None

    @property
    def is_open(self) -> bool:
        """Whether the index is currently open."""
        return self._index is not None

    def open(self) -> None:
        """Create a fresh, empty in-memory vector index.

        If the index is already open, it is closed first.
        """
        if self.is_open:
            self.close()

        self._index = VectorSearch(keyword_fields=["course"])

    def reset(self) -> None:
        """No-op reset for the volatile in-memory vector index.

        The index is never persisted and ``open()`` already discards any
        previous content by creating a new empty ``VectorSearch``, so there is
        nothing to clear. The method exists only to honour the
        ``VectorIndex`` contract invoked by ``SemanticalFAQ.build_index``.
        """
        pass

    def load(
        self, documents: Sequence[dict[str, str]], vectors: Sequence[Vector]
    ) -> None:
        """Replace the store contents with the given documents and embeddings.

        The previous content is discarded; the store is fully rebuilt from the
        provided data.

        Args:
            documents: The documents to store.
            vectors: The embeddings of the documents, aligned with ``documents``.

        Raises:
            ValueError: If ``documents`` and ``vectors`` have different lengths.
            RuntimeError: If the index is closed and cannot be opened.
        """
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")

        if self._index is None:
            self.open()
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        index.fit(list(vectors), list(documents))

    def search(
        self,
        query_vector: Vector,
        filter_dict: dict[str, str] | None = None,
        num_results: int = 5,
    ) -> list[SearchResult]:
        """Search the store by vector similarity.

        Args:
            query_vector: The embedding of the query.
            filter_dict: Optional mapping of field names to values used to
                restrict the search to matching documents.
            num_results: Maximum number of results to return.

        Returns:
            The matching results, ordered from closest to farthest.

        Raises:
            RuntimeError: If the index is closed.
        """
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)

        normalized = normalize_index_param(filter_dict)

        rows = index.search(
            query_vector, filter_dict=normalized, num_results=num_results
        )
        return [
            SearchResult(document=row, distance=0.0, rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def size(self) -> int:
        """Return the number of stored documents.

        Returns:
            The document count.

        Raises:
            RuntimeError: If the index is closed.
        """
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        return len(index.docs)

    def close(self) -> None:
        """Close the index and release managed resources."""
        self._index = None
        self._release_resource()
