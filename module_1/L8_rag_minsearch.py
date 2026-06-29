from module_1.classes.faq_searcher_factory import FAQSearcherFactory
from module_1.classes.llm_instructions import PipelineLLMInstruction
from module_1.classes.llm_prompter import LLMPrompter
from module_1.classes.prompt_builder import PromptBuilder
from module_1.classes.rag_pipeline import RAGPipeline
from module_1.classes.searcher_config import SearcherConfig
from module_1.classes.thread import Thread

search_config = SearcherConfig(
    index_type="in_memory",
)

searcher_factory = FAQSearcherFactory()
searcher = searcher_factory.create(search_config)

thread = Thread(PipelineLLMInstruction())
prompt_builder = PromptBuilder(thread)

llm_prompter = LLMPrompter()

rag = RAGPipeline(searcher, prompt_builder, llm_prompter)

question = "The course has begun. Could I enroll anyway?"

answer = rag.answer(question)

print(answer)
