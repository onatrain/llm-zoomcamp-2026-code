"""SQLite-backed vector index using ``sqlitesearch.VectorSearchIndex``."""

from typing import Sequence

from sqlitesearch import VectorSearchIndex

from classes.managed_resource import ManagedResource
from classes.vector_index import Vector, VectorIndex
from utils.normalize import normalize_index_param


class LocalVectorDbIndex(ManagedResource[VectorSearchIndex], VectorIndex):
    """Vector index persisted in SQLite via sqlitesearch.

    The store is persistent but treated as read-only and immutable: its content
    is replaced wholesale through ``reset()`` followed by ``load()`` rather than
    appended to incrementally.
    """

    closed_message = "Index closed. You must open() it first."

    def __init__(self, db_path: str = "./vector_faq.db") -> None:
        """Initialize a closed index backed by the given SQLite file.

        Args:
            db_path: Path to the SQLite database file.
        """
        super().__init__()
        self._db_path = db_path

    def open(self) -> None:
        """Open (or reopen) the connection to the SQLite store.

        If the index is already open, it is closed first.

        Raises:
            RuntimeError: If the underlying store cannot be opened.
        """
        if self.is_open:
            self.close()

        index = VectorSearchIndex(
            keyword_fields=["course"],
            mode="ivf",
            db_path=self._db_path,
        )
        self._register_resource(index, index.close)

    def reset(self) -> None:
        """Destroy all stored content so the index can be rebuilt.

        The index must already be open; this method never opens a connection
        itself. The caller is responsible for opening (and closing) the index.

        Raises:
            RuntimeError: If the index is closed.
        """
        index = self._require_resource()
        index.clear()

    def load(
        self, documents: Sequence[dict[str, str]], vectors: Sequence[Vector]
    ) -> None:
        """Load documents and their embeddings into the index.

        Assumes the index is empty; call ``reset()`` first to clear it. Opens
        the index if it is closed.

        Args:
            documents: Documents to fit into the index.
            vectors: Embeddings aligned with ``documents``.

        Raises:
            ValueError: If ``documents`` and ``vectors`` have different lengths.
            RuntimeError: If the index is closed and cannot be opened.
        """
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")

        if not self.is_open:
            self.open()
        index = self._require_resource()
        if not documents:
            return
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
        return int(self._require_resource().count())

    def close(self) -> None:
        """Close the index and release managed resources."""
        resource = self._resource
        if resource is not None:
            resource.close()
        self._release_resource()
