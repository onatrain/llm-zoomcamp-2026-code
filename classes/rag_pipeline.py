from typing import Any

from classes.faq_searcher import FAQSearcher
from classes.llm_prompter import LLMPrompter
from classes.prompt_builder import PromptBuilder


class RAGPipeline:
    def __init__(
        self,
        searcher: FAQSearcher,
        prompt_builder: PromptBuilder,
        llm_prompter: LLMPrompter,
    ):
        self._searcher = searcher
        self._prompt_builder = prompt_builder
        self._llm_prompter = llm_prompter

    def answer(
        self,
        question: str,
        course: str = "llm-zoomcamp",
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        boost_dict = {
            "question": 2.0,
            "section": 1.0,
            "answer": 0.5,
        }
        merged_extra: dict[str, Any] = {"boost_dict": boost_dict}
        if extra_params:
            merged_extra.update(extra_params)

        retrieval = self._searcher.search(
            question, course=course, extra_params=merged_extra
        )

        thread = self._prompt_builder.build(
            question, retrieval
        )  # crea el user prompt y lo anexa al hilo y lo retorna

        generation = self._llm_prompter.generate(
            thread
        )  # recibe y trabaja con el hilo actual

        return generation


"""
En answer:
1) consultar la db
2) construir el user prompt y obtener el hilo actual (system + user)
3) generar texto usando el hilo actual
"""
