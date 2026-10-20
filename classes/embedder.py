"""Port for text embedders used to vectorize FAQ documents."""

from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class Embedder(ABC):
    """Contract for text embedders.

    Implementations own the model-specific details (tokenization, batching,
    normalization) and expose them through this vocabulary.
    """

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Dimension of the embeddings produced by this embedder.

        Returns:
            The number of components of every embedding vector.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> NDArray[Any]:
        """Embed a list of texts.

        Args:
            texts: The texts to embed.

        Returns:
            A 2-D array with one embedding per text.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str) -> NDArray[Any]:
        """Embed a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding of the text as a 1-D array.
        """
        raise NotImplementedError
