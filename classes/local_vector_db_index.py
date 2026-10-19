"""SQLite-backed vector index using ``sqlitesearch.VectorSearchIndex``."""

from typing import Sequence

from sqlitesearch import VectorSearchIndex

from classes.managed_resource import ManagedResource
from classes.vector_index import SearchResult, Vector, VectorIndex
from utils.normalize import normalize_index_param


class LocalVectorDbIndex(ManagedResource[VectorSearchIndex], VectorIndex):
    """Vector index persisted in SQLite via sqlitesearch."""

    closed_message = "Index closed. You must open() it first."

    def __init__(self, db_path: str = "./vector_faq.db") -> None:
        super().__init__()
        self._db_path = db_path
        self._index: VectorSearchIndex | None = None

    @property
    def is_open(self) -> bool:
        return self._index is not None

    def open(self) -> None:
        if self.is_open:
            self.close()

        index = VectorSearchIndex(
            keyword_fields=["course"],
            mode="ivf",
            db_path=self._db_path,
        )
        self._index = index
        self._register_resource(index, index.close)

    def reset(self) -> None:
        if self._index is None:
            self.open()
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        index.clear()

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

        if len(documents) == 0:
            index.clear()
            return

        index.fit(list(vectors), list(documents))

    def search(
        self,
        query_vector: Vector,
        filter_dict: dict[str, str] | None = None,
        num_results: int = 5,
    ) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError(self.closed_message)

        index = self._index
        normalized = normalize_index_param(filter_dict)
        rows = index.search(
            query_vector, filter_dict=normalized, num_results=num_results
        )

        results: list[SearchResult] = []
        for rank, row in enumerate(rows, start=1):
            if isinstance(row, dict):
                distance = float(row.get("distance", 0.0))
                document = row
            else:
                distance = 0.0
                document = dict(row) if hasattr(row, "items") else {"value": str(row)}
            results.append(
                SearchResult(document=document, distance=distance, rank=rank)
            )

        return results

    def size(self) -> int:
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        return int(index.count())

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
        self._index = None
        self._release_resource()
