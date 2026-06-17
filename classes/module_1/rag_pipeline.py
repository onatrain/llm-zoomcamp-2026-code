from classes.module_1.faq_searcher import FAQSearcher
from classes.module_1.llm_prompter import LLMPrompter
from classes.module_1.prompt_builder import PromptBuilder


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

        prompt = self._prompt_builder.build(question, retrieval)

        generation = self._llm_prompter.generate(prompt)

        return generation
