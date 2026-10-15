from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from numpy.typing import NDArray

Vector = NDArray[Any]


@dataclass(frozen=True)
class SearchResult:
    """A single hit returned by a vector store search.

    Attributes:
        document: The stored document as a string mapping.
        distance: The distance between the query vector and the document
            vector, in the metric used by the store.
        rank: The 1-based position of the result in the result list.
    """

    document: dict[str, str]
    distance: float
    rank: int


class VectorStore(ABC):
    """Contract for vector database stores used to search embedded documents.

    Implementations own all the technology-specific details (connections, SQL
    dialects, index maintenance) and expose them through this vocabulary.
    """

    closed_message = "Store closed. You must connect() it first."

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension expected by this store."""
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> None:
        """Open the connection to the store."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the connection to the store is open.

        Returns:
            True if the store is connected, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def db_reset(self) -> None:
        """Drop and recreate the store schema.

        This operation is destructive: it deletes every stored document.
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, document: dict[str, str], vector: Vector) -> None:
        """Insert a single document with its embedding.

        Args:
            document: The document fields to store.
            vector: The embedding of the document.
        """
        raise NotImplementedError

    @abstractmethod
    def bulk_insert(
        self, documents: Sequence[dict[str, str]], vectors: Sequence[Vector]
    ) -> None:
        """Insert many documents with their embeddings.

        Args:
            documents: The documents to store.
            vectors: The embeddings of the documents, aligned with ``documents``.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_vector: Vector, course: str | None = None, num_results: int = 5
    ) -> list[SearchResult]:
        """Search the store by vector similarity.

        Args:
            query_vector: The embedding of the query.
            course: Optional course slug to restrict the search to.
            num_results: Maximum number of results to return.

        Returns:
            The matching results, ordered from closest to farthest.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored documents.

        Returns:
            The document count.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the store."""
        raise NotImplementedError
