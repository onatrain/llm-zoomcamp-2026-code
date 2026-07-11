from minsearch import VectorSearch

from classes.faq import FAQ
from classes.raw_faq import RawFaq
from classes.embedder import Embedder


class InMemoryVectorFAQ(FAQ):
    def __init__(self, embedder: Embedder) -> None:
        self._model = embedder
        self._index = None

    def open(self) -> None:
        if not isinstance(self._index, InMemoryVectorFAQ):
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

        query_vector = self._model.encode_text(question)

        results: list[dict[str, str]] = self._index.search(
            query_vector,
            filter_dict=filter_dict,
            num_results=num_results
        )

        return results

    def close(self) -> None:
        self._index = None

    def build_index(self) -> None:
        faq = RawFaq()
        faq_items = faq.items

        texts = [" ".join((d["question"], d["answer"])) for d in faq_items]

        embeddings = self._model.embed_texts(texts)

        index = VectorSearch(
            keyword_fields=['course'],
        )

        index.fit(embeddings, faq_items)

        self._index = index

    @property
    def count(self) -> int:
        if self._index is None:
            raise RuntimeError("FAQ closed. You must open it first.")

        return len(self._index.vectors)
