from abc import ABC
from typing import Any

from numpy.typing import NDArray


class Embedder(ABC):
    @property
    def embedding_dimension(self) -> int:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> NDArray[Any]:
        raise NotImplementedError

    def encode_text(self, text: str) -> NDArray[Any]:
        raise NotImplementedError
