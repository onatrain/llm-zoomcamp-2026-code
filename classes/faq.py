from abc import ABC, abstractmethod
from typing import Any


class FAQ(ABC):
    """Contract for an FAQ corpus that can be searched and rebuilt."""

    @abstractmethod
    def open(self) -> None:
        """Open the FAQ source."""
        raise NotImplementedError

    @abstractmethod
    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        """Search the FAQ content.

        The accepted parameters are defined by each concrete adapter.

        Returns:
            The matching documents, ordered from closest to farthest.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the FAQ source."""
        raise NotImplementedError

    @abstractmethod
    def build_index(self) -> None:
        """Rebuild the index with the current FAQ content.

        Implementations must keep the previous content untouched when the new
        content cannot be fetched or embedded.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of documents currently indexed.

        Returns:
            The document count.
        """
        raise NotImplementedError
