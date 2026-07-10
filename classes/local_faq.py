import weakref
from typing import Any

from sqlitesearch import TextSearchIndex

from classes.faq import FAQ
from classes.raw_faq import RawFaq


class LocalFAQ(FAQ):
    def __init__(self, db_path: str = "./faq.db"):
        self._db_path = db_path
        self._index: TextSearchIndex | None = None
        self._finalizer: weakref.finalize[Any, Any] | None = None

    def open(self) -> None:
        if self._index is not None:
            self.close()

        index = TextSearchIndex(
            text_fields=['question', 'section', 'answer'],
            keyword_fields=['course'],
            db_path=self._db_path
        )

        self._finalizer = weakref.finalize(self, index.close)

        self._index = index

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[dict[str, str]]:

        if self._index is None:
            raise RuntimeError("FAQ closed. You must open it first.")

        results: list[dict[str, str]] = self._index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results
        )

        return results

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
        self._index = None

        # Must detach finalizer to avoid double calling when GC cleans later.
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer.detach()
            self._finalizer = None

    def build_index(self) -> None:
        raw_faq = RawFaq()
        faq_items = raw_faq.items

        self.open()
        if self._index is not None:
            self._index.fit(faq_items)

    @property
    def count(self) -> int:
        if self._index is None:
            raise RuntimeError("FAQ closed. You must open it first.")

        item_count: int = self._index.count()
        return item_count
