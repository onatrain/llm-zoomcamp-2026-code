import weakref
from typing import Any

from sqlitesearch import TextSearchIndex  # type: ignore

from classes.module_1.faq import FAQ
from classes.module_1.raw_faq import RawFaq


class LocalFAQ(FAQ):
    def __init__(self, db_path: str = "./faq.db"):
        self._db_path = db_path
        self._index: TextSearchIndex | None = None
        self._finalizer: weakref.finalize[Any, Any] | None = None

    def open(self) -> TextSearchIndex:
        if self._index is not None:
            self.close()

        index = TextSearchIndex(
            text_fields=['question', 'section', 'answer'],
            keyword_fields=['course'],
            db_path=self._db_path
        )

        self._finalizer = weakref.finalize(self, index.close)

        self._index = index

        return index

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

    def build_index(self) -> TextSearchIndex:
        raw_faq = RawFaq()
        faq_items = raw_faq.items

        index = self.open()
        index.fit(faq_items)

        return index
