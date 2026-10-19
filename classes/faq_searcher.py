from typing import Any

from classes.faq import FAQ


class FAQSearcher:
    def __init__(self, index: FAQ):
        self._index = index

    def search(
        self,
        question: str,
        course: str = "llm-zoomcamp",
        extra_params: dict[str, Any] | None = None,
        num_results: int = 5,
    ) -> list[dict[str, str]]:
        return self._index.search(
            question,
            filter_dict={"course": course},
            num_results=num_results,
            extra_params=extra_params,
        )
