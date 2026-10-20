"""In-memory vector index backed by ``minsearch.VectorSearch``."""

from functools import partial
from typing import Sequence

from minsearch import VectorSearch

from classes.managed_resource import ManagedResource
from classes.vector_index import Vector, VectorIndex
from utils.normalize import normalize_index_param


def _clear_vector_index(index: VectorSearch) -> None:
    """Release the heavy buffers held by a minsearch VectorSearch.

    This smell is caused by the lack of explicit memory cleaning in
    minsearch.VectorSearch

    Args:
        index: The index whose internal buffers must be cleared.
    """
    index.vectors = None
    index.docs = []
    index.keyword_df = None
    index.numeric_df = None
    index.date_df = None
    index._filter = None


class InMemoryVectorIndex(ManagedResource[VectorSearch], VectorIndex):
    """In-memory vector index using cosine similarity via minsearch.

    The index is volatile: its state is never persisted, so opening it always
    yields a fresh, empty instance. ``reset()`` is therefore a no-op that
    exists only to satisfy the ``VectorIndex`` contract.

    The underlying ``minsearch.VectorSearch`` has no ``close()`` method, so a
    closer that clears its heavy buffers is registered with
    ``ManagedResource`` and runs both on explicit ``close()`` and on garbage
    collection.
    """

    closed_message = "Index closed. You must open() it first."

    def __init__(self) -> None:
        """Initialize a closed, empty in-memory vector index."""
        super().__init__()

    def open(self) -> None:
        """Create a fresh, empty in-memory vector index.

        If the index is already open, it is closed first.
        """
        if self.is_open:
            self.close()

        index = VectorSearch(keyword_fields=["course"])
        self._register_resource(index, partial(_clear_vector_index, index))

    def reset(self) -> None:
        """No-op reset for the volatile in-memory vector index.

        The index is never persisted and ``open()`` already discards any
        previous content by creating a new empty ``VectorSearch``, so there is
        nothing to clear. The method exists only to honour the
        ``VectorIndex`` contract invoked by ``SemanticalFAQ.build_index``.
        """
        # Volatile index: ``open()`` already discards any previous content by
        # creating a brand-new, empty ``VectorSearch``. There is nothing to
        # clear here; the method only honours the VectorIndex contract.
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

        if not self.is_open:
            self.open()
        index = self._require_resource()
        index.fit(list(vectors), list(documents))

    def search(
        self,
        query_vector: Vector,
        filter_dict: dict[str, str] | None = None,
        num_results: int = 5,
    ) -> list[dict[str, str]]:
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
        index = self._require_resource()
        normalized = normalize_index_param(filter_dict)

        rows: list[dict[str, str]] = index.search(
            query_vector, filter_dict=normalized, num_results=num_results
        )
        return rows

    def size(self) -> int:
        """Return the number of stored documents.

        Returns:
            The document count.

        Raises:
            RuntimeError: If the index is closed.
        """
        return len(self._require_resource().docs)

    def close(self) -> None:
        """Close the index and release managed resources."""
        resource = self._resource
        if resource is not None:
            _clear_vector_index(resource)
        self._release_resource()
