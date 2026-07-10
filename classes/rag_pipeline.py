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

    def answer(self, question: str) -> str:
        retrieval = self._searcher.search(question)

        thread = self._prompt_builder.build(question, retrieval)  # crea el user prompt y lo anexa al hilo y lo retorna

        generation = self._llm_prompter.generate(thread)  # recibe y trabaja con el hilo actual

        return generation

"""
En answer:
1) consultar la db
2) construir el user prompt y obtener el hilo actual (system + user)
3) generar texto usando el hilo actual
"""