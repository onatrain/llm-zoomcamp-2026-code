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
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError
