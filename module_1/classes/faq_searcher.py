from module_1.classes.faq import FAQ


class FAQSearcher:
    def __init__(self, index: FAQ):
        self._index = index

    def search(self, question: str, course: str = 'llm-zoomcamp') -> list[dict[str, str]]:
        # Con boost_dict se puede indicar cuál campo tiene más peso en el ranking
        return self._index.search(
            question,
            boost_dict={'question': 2.0, 'section': 1.0, 'answer': 0.5},  # default boost = 1
            filter_dict={'course': course},
            num_results=5
        )
