from abc import ABC, abstractmethod
from typing import Any, Sequence

from numpy.typing import NDArray

Vector = NDArray[Any]


class VectorIndex(ABC):
    """Contract for vector database stores used to search embedded documents.

    Implementations own all the technology-specific details (connections, SQL
    dialects, index maintenance) and expose them through this vocabulary.
    """

    closed_message = "Index closed. You must connect() it first."

    @abstractmethod
    def open(self) -> None:
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
    def reset(self) -> None:
        """Drop and recreate the store schema.

        This operation is destructive: it deletes every stored document.
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        """Return the number of stored documents.

        Returns:
            The document count.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the store."""
        raise NotImplementedError
