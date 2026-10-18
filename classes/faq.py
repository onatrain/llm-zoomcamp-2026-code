from abc import ABC, abstractmethod
from typing import Any


class FAQ(ABC):
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_index(self) -> None:
        """Rebuild the index with the current FAQ content.

        Implementations must keep the previous content untouched when the new
        content cannot be fetched or embedded.
        """
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError
