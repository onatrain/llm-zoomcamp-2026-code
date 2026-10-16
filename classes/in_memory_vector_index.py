"""In-memory vector index backed by ``minsearch.VectorSearch``."""

from typing import Sequence

from minsearch import VectorSearch

from classes.managed_resource import ManagedResource
from classes.vector_index import SearchResult, Vector, VectorIndex


class InMemoryVectorIndex(ManagedResource[VectorSearch], VectorIndex):
    """In-memory vector index using cosine similarity via minsearch."""

    closed_message = "Index closed. You must open() it first."

    def __init__(self) -> None:
        super().__init__()
        self._index: VectorSearch | None = None
        self._documents: list[dict[str, str]] = []
        self._vectors: list[Vector] = []

    @property
    def is_open(self) -> bool:
        return self._index is not None

    def open(self) -> None:
        if self.is_open:
            self.close()

        self._documents = []
        self._vectors = []
        self._index = VectorSearch(keyword_fields=["course"])

    def reset(self) -> None:
        self.close()
        self.open()

    def add(self, document: dict[str, str], vector: Vector) -> None:
        self.add_all([document], [vector])

    def add_all(
        self, documents: Sequence[dict[str, str]], vectors: Sequence[Vector]
    ) -> None:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")

        self._documents.extend(documents)
        self._vectors.extend(vectors)

        if not self.is_open:
            self.open()

        self._index = VectorSearch(keyword_fields=["course"])
        self._index.fit(self._vectors, self._documents)

    def search(
        self, query_vector: Vector, course: str | None = None, num_results: int = 5
    ) -> list[SearchResult]:
        if not self.is_open or self._index is None:
            raise RuntimeError(self.closed_message)

        filter_dict = {"course": course} if course else None
        rows = self._index.search(
            query_vector, filter_dict=filter_dict, num_results=num_results
        )
        return [
            SearchResult(document=row, distance=0.0, rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def size(self) -> int:
        if not self.is_open or self._index is None:
            raise RuntimeError(self.closed_message)
        return len(self._documents)

    def close(self) -> None:
        self._index = None
        self._documents = []
        self._vectors = []
