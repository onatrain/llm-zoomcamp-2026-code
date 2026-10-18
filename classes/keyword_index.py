"""Port for lexical (keyword) indexes used to search FAQ documents."""

from abc import ABC, abstractmethod
from typing import Sequence


class KeywordIndex(ABC):
    """Contract for keyword (text) indexes."""

    closed_message = "Index closed. You must open() it first."

    @abstractmethod
    def open(self) -> None:
        """Open the index."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the index is open."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear the index for a full rebuild."""
        raise NotImplementedError

    @abstractmethod
    def load(self, documents: Sequence[dict[str, str]]) -> None:
        """Load many documents into the index (populate it)."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[dict[str, str]]:
        """Search the index."""
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        """Return the number of stored documents."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the index."""
        raise NotImplementedError
