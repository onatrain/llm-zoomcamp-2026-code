from classes.module_1.faq_searcher_factory import FAQSearcherFactory
from classes.module_1.llm_prompter import LLMPrompter
from classes.module_1.prompt_builder import PromptBuilder
from classes.module_1.rag_pipeline import RAGPipeline
from classes.module_1.searcher_config import SearcherConfig

search_config = SearcherConfig(
    index_type="in_memory",
)

searcher_factory = FAQSearcherFactory()
searcher = searcher_factory.create(search_config)

prompt_builder = PromptBuilder()

llm_prompter = LLMPrompter()

rag = RAGPipeline(searcher, prompt_builder, llm_prompter)

question = "The course has begun. Could I enroll anyway?"

answer = rag.answer(question)

print(answer)
