"""Lexical FAQ adapter for any keyword index."""

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
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[dict[str, str]]:
        return self._index.search(
            question,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=num_results,
        )

    def close(self) -> None:
        self._index.close()

    def build_index(self) -> None:
        self.open()
        self._index.reset()
        self._index.add_all(RawFaq().items)

    @property
    def count(self) -> int:
        return self._index.size()
