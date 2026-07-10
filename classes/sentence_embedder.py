import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from classes.embedder import Embedder


class SentenceEmbedder(Embedder):
    def __init__(self) -> None:
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = './models'
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2", token=False)

    def embed_texts(self, texts: list[str]) -> NDArray[Any]:
        batch_size = 50

        batch_generator = (
            self._model.encode(texts[i : i + batch_size])
            for i in tqdm(range(0, len(texts), batch_size), desc="Procesando batches")
        )

        vectors = [vector for batch in batch_generator for vector in batch]

        return np.array(vectors)

    def encode_text(self, text: str) -> NDArray[Any]:
        result: NDArray[Any] = self._model.encode(text, convert_to_numpy=True)
        return result
