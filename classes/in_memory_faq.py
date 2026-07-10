from minsearch import Index

from classes.faq import FAQ
from classes.raw_faq import RawFaq


class InMemoryFAQ(FAQ):
    def __init__(self) -> None:
        self._index = None

    def open(self) -> None:
        if not isinstance(self._index, InMemoryFAQ):
            self.build_index()

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10
    ) -> list[dict[str, str]]:

        if self._index is None:
            raise RuntimeError("FAQ closed. You must open it first.")

        # Con boost_dict se puede indicar cuál campo tiene más peso en el ranking
        results: list[dict[str, str]] = self._index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results
        )

        return results

    def close(self) -> None:
        self._index = None

    def build_index(self) -> None:
        faq = RawFaq()
        faq_items = faq.items

        index = Index(
            text_fields=['question', 'section', 'answer'],
            keyword_fields=['course'],
        )

        index.fit(faq_items)

        self._index = index

    @property
    def count(self) -> int:
        if self._index is None:
            raise RuntimeError("FAQ closed. You must open it first.")

        return self._index.count()
