from pathlib import Path

from classes.faq_searcher_factory import FAQSearcherFactory
from classes.llm_instructions import PipelineLLMInstruction
from classes.llm_prompter import LLMPrompter
from classes.prompt_builder import PromptBuilder
from classes.rag_pipeline import RAGPipeline
from classes.searcher_config import SearcherConfig
from classes.thread import Thread

current_dir = Path(__file__).parent

search_config = SearcherConfig(
    index_type="local",
    vector_engine="sentence",
    local_index_path=str(current_dir / "data/vector_faq.db"),
)

searcher_factory = FAQSearcherFactory()
searcher = searcher_factory.create(search_config)

thread = Thread(PipelineLLMInstruction())
prompt_builder = PromptBuilder(thread)

llm_prompter = LLMPrompter()

rag = RAGPipeline(searcher, prompt_builder, llm_prompter)

question = "Will I obtain a completion certificate after I finish the course?"

answer = rag.answer(question)

print(answer)
