"""SQLite-backed keyword index using ``sqlitesearch.TextSearchIndex``."""

from typing import Sequence

from sqlitesearch import TextSearchIndex

from classes.managed_resource import ManagedResource
from classes.keyword_index import KeywordIndex


class LocalDbIndex(ManagedResource[TextSearchIndex], KeywordIndex):
    """Keyword index persisted in SQLite via sqlitesearch."""

    closed_message = "Index closed. You must open() it first."

    def __init__(self, db_path: str = "./faq.db") -> None:
        super().__init__()
        self._db_path = db_path
        self._index: TextSearchIndex | None = None

    @property
    def is_open(self) -> bool:
        return self._index is not None

    def open(self) -> None:
        if self.is_open:
            self.close()

        index = TextSearchIndex(
            text_fields=["question", "section", "answer"],
            keyword_fields=["course"],
            db_path=self._db_path,
        )
        self._index = index
        self._register_resource(index, index.close)

    def reset(self) -> None:
        if self._index is None:
            self.open()
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        index.clear()

    def add(self, document: dict[str, str]) -> None:
        if self._index is None:
            self.open()
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        index.add(document)

    def add_all(self, documents: Sequence[dict[str, str]]) -> None:
        if self._index is None:
            self.open()
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        if not documents:
            index.clear()
            return
        index.fit(list(documents))

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
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        return int(index.count())

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
        self._index = None
        self._release_resource()
