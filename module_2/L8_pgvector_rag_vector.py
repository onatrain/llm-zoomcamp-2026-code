import dotenv
import os

from classes.faq_searcher_factory import FAQSearcherFactory
from classes.llm_instructions import PipelineLLMInstruction
from classes.llm_prompter import LLMPrompter
from classes.prompt_builder import PromptBuilder
from classes.rag_pipeline import RAGPipeline
from classes.searcher_config import SearcherConfig
from classes.thread import Thread

dotenv.load_dotenv()

search_config = SearcherConfig(
    index_type="repository",
    vector_engine="sentence",
    database_url=os.getenv("DATABASE_URL"),
)

searcher_factory = FAQSearcherFactory()
searcher = searcher_factory.create(search_config)

thread = Thread(PipelineLLMInstruction())
prompt_builder = PromptBuilder(thread)

llm_prompter = LLMPrompter()

rag = RAGPipeline(searcher, prompt_builder, llm_prompter)

# question = "Will I obtain a completion certificate after I finish the course?"
# question = "Will I have to do some homework?"
# question = "How much time will I have to dedicate to the course weekly?"
question = "Can I use the workshop materials for my own creations?"

answer = rag.answer(question)

print(answer)
