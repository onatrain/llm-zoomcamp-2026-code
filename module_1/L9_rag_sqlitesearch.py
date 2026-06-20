from pathlib import Path

from classes.module_1.faq_searcher_factory import FAQSearcherFactory
from classes.module_1.llm_prompter import LLMPrompter
from classes.module_1.prompt_builder import PromptBuilder
from classes.module_1.rag_pipeline import RAGPipeline
from classes.module_1.searcher_config import SearcherConfig
from classes.module_1.llm_instructions import PipelineLLMInstruction
from classes.module_1.thread import Thread


current_dir = Path(__file__).parent

search_config = SearcherConfig(
    index_type="local",
    local_index_path=str(current_dir / "data/faq.db"),
)

searcher_factory = FAQSearcherFactory()
searcher = searcher_factory.create(search_config)

thread = Thread(PipelineLLMInstruction())
prompt_builder = PromptBuilder(thread)

llm_prompter = LLMPrompter()

rag = RAGPipeline(searcher, prompt_builder, llm_prompter)

question = "Is the course really free?"

answer = rag.answer(question)

print(answer)
