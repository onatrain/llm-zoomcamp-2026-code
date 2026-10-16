"""In-memory keyword index backed by ``minsearch.Index``."""

from typing import Sequence

from minsearch import Index

from classes.managed_resource import ManagedResource
from classes.keyword_index import KeywordIndex


class InMemoryIndex(ManagedResource[Index], KeywordIndex):
    """In-memory keyword index using minsearch."""

    closed_message = "Index closed. You must open() it first."

    def __init__(self) -> None:
        super().__init__()
        self._index: Index | None = None
        self._documents: list[dict[str, str]] = []

    @property
    def is_open(self) -> bool:
        return self._index is not None

    def open(self) -> None:
        if self.is_open:
            self.close()

        self._index = Index(
            text_fields=["question", "section", "answer"],
            keyword_fields=["course"],
        )

    def reset(self) -> None:
        self.close()
        self._documents = []
        self.open()

    def add(self, document: dict[str, str]) -> None:
        self.add_all([document])

    def add_all(self, documents: Sequence[dict[str, str]]) -> None:
        self._documents.extend(documents)

        if not self.is_open:
            self.open()

        if self._documents:
            self._index = Index(
                text_fields=["question", "section", "answer"],
                keyword_fields=["course"],
            )
            self._index.fit(self._documents)

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[dict[str, str]]:
        if not self.is_open or self._index is None:
            raise RuntimeError(self.closed_message)

        results = self._index.search(
            question,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=num_results,
        )
        return list(results)

    def size(self) -> int:
        if not self.is_open or self._index is None:
            raise RuntimeError(self.closed_message)
        return len(self._documents)

    def close(self) -> None:
        self._index = None
        self._documents = []
