from classes.faq_searcher_factory import FAQSearcherFactory
from classes.llm_instructions import PipelineLLMInstruction
from classes.llm_prompter import LLMPrompter
from classes.prompt_builder import PromptBuilder
from classes.rag_pipeline import RAGPipeline
from classes.searcher_config import SearcherConfig
from classes.thread import Thread

search_config = SearcherConfig(
    index_type="in_memory",
    vector_engine="sentence",
)

searcher_factory = FAQSearcherFactory()
searcher = searcher_factory.create(search_config)

print(searcher._index.count)

thread = Thread(PipelineLLMInstruction())
prompt_builder = PromptBuilder(thread)

llm_prompter = LLMPrompter()

rag = RAGPipeline(searcher, prompt_builder, llm_prompter)

question = "The course has begun. Could I enroll anyway?"

answer = rag.answer(question)

print(answer)
