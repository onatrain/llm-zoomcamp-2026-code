from abc import ABC, abstractmethod
from typing import Any


class FAQ(ABC):
    @abstractmethod
    def open(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_index(self) -> Any:
        raise NotImplementedError
