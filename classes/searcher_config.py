from dataclasses import dataclass
from typing import Literal


@dataclass
class SearcherConfig:
    index_type: Literal["in_memory", "local"]  # "repository"
    vector_engine: Literal["sentence"] | None = None
    local_index_path: str | None = None
