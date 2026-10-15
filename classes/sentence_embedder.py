import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from classes.embedder import Embedder

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_MODEL_DIR = Path(MODEL_ID.split("/")[-1])
_CACHE_REPO = "models--" + MODEL_ID.replace("/", "--")

# huggingface_hub and transformers read these flags once, at import time, so
# they must be set before any huggingface library is imported. They are only
# defaults: an already set value is preserved.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def hub_cache_dir() -> Path:
    """Return the Hugging Face Hub cache directory in use.

    Returns:
        The path of the directory where the Hub caches model repositories.
    """
    hub_cache = os.getenv("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()

    hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
    return Path(hf_home).expanduser() / "hub"


def _snapshot_dirs(cache_dir: Path) -> list[Path]:
    """List the cached snapshot directories of this model under a cache dir.

    Args:
        cache_dir: A Hugging Face cache directory to inspect.

    Returns:
        The snapshot directories found, most recent first.
    """
    snapshots = cache_dir.glob(f"{_CACHE_REPO}/snapshots/*")
    return sorted((path for path in snapshots if path.is_dir()), reverse=True)


def local_model_path() -> Path | None:
    """Find the model directory already present in this machine.

    `SENTENCE_TRANSFORMERS_HOME` is inspected first, then the Hugging Face Hub
    cache. When a local directory is found, no Hub lookup is ever performed.

    Returns:
        The local directory containing the model, or None if it is not cached.
    """
    st_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    search_dirs: list[Path] = []

    if st_home:
        st_home_path = Path(st_home).expanduser()
        search_dirs.append(st_home_path)

        # Legacy sentence-transformers layout: <home>/<model-name>
        legacy_dir = st_home_path / _MODEL_DIR
        if (legacy_dir / "modules.json").is_file():
            return legacy_dir

    search_dirs.append(hub_cache_dir())

    for cache_dir in search_dirs:
        for snapshot in _snapshot_dirs(cache_dir):
            if (snapshot / "modules.json").is_file():
                return snapshot

    return None


def load_transformer() -> Any:
    """Load the sentence transformer without reaching the network.

    Returns:
        The loaded `SentenceTransformer` model.

    Raises:
        RuntimeError: If the model is not cached locally, since offline mode
            prevents downloading it.
    """
    from sentence_transformers import SentenceTransformer

    model_path = local_model_path()
    if model_path is None:
        raise RuntimeError(
            f"Model {MODEL_ID} is not available in the local cache and offline "
            f"mode is enabled. Download it once with `huggingface-cli download "
            f"{MODEL_ID}` and try again."
        )

    return SentenceTransformer(str(model_path), token=False)


class SentenceEmbedder(Embedder):
    """Embedder backed by a local sentence-transformers model."""

    def __init__(self) -> None:
        self._model = load_transformer()
        self._embedding_dimension: int | None = None

    def embed_texts(self, texts: list[str]) -> NDArray[Any]:
        """Embed a list of texts in batches.

        Args:
            texts: The texts to embed.

        Returns:
            A 2-D array with one embedding per text.
        """
        batch_size = 50

        batch_generator = (
            self._model.encode(texts[i : i + batch_size])
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches")
        )

        vectors = [vector for batch in batch_generator for vector in batch]

        return np.array(vectors)

    @property
    def embedding_dimension(self) -> int:
        """Dimension of the embeddings produced by this embedder.

        Returns:
            Number of components of every embedding vector.
        """
        if self._embedding_dimension is None:
            self._embedding_dimension = int(self.encode_text("probe").shape[0])

        return self._embedding_dimension

    def encode_text(self, text: str) -> NDArray[Any]:
        """Embed a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding of the text as a 1-D array.
        """
        result: NDArray[Any] = self._model.encode(text, convert_to_numpy=True)
        return result
