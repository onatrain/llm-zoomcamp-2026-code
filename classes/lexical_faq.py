"""Lexical FAQ adapter for any keyword index."""

from typing import Any

from classes.faq import FAQ
from classes.keyword_index import KeywordIndex
from classes.raw_faq import RawFaq


class LexicalFAQ(FAQ):
    """FAQ adapter that searches documents through a keyword index."""

    def __init__(self, index: KeywordIndex) -> None:
        self._index = index

    def open(self) -> None:
        if not self._index.is_open:
            self._index.open()

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        num_results: int = 5,
        boost_dict: dict[str, float] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        merged_extra: dict[str, Any] = dict(extra_params or {})
        if boost_dict is not None:
            merged_extra["boost_dict"] = boost_dict

        return self._index.search(
            question,
            filter_dict=filter_dict,
            num_results=num_results,
            **merged_extra,
        )

    def close(self) -> None:
        self._index.close()

    def build_index(self) -> None:
        """Rebuild the index from the current FAQ content.

        The previous content is only replaced once the new content has been
        fetched, so a failed download leaves the index untouched.

        Raises:
            RuntimeError: If the FAQ source returns no documents.
        """
        faq_items = RawFaq().items
        if not faq_items:
            raise RuntimeError(
                "The FAQ source returned no documents; the index was not modified"
            )

        self.open()

        self._index.reset()
        self._index.load(faq_items)

    @property
    def count(self) -> int:
        return self._index.size()
