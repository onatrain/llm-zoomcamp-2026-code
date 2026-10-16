from pathlib import Path

from classes.agent import Agent
from classes.lexical_faq import LexicalFAQ
from classes.llm_instructions import SearchAgentLLMInstruction
from classes.llm_prompter import LLMPrompter
from classes.local_db_index import LocalDbIndex
from classes.thread import Thread
from classes.toolset import Toolset

current_dir = Path(__file__).parent
index_path = str(current_dir / "data/faq.db")

faq = LexicalFAQ(LocalDbIndex(index_path))
faq.open()


def search(query: str) -> list[dict[str, str]]:
    """Search the FAQ database with boosted and filtered fields.

    Args:
        query: The search query text.

    Returns:
        A list of matching FAQ entries as dictionaries.
    """
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return faq.search(query, filter_dict=filter_dict, boost_dict=boost_dict, num_results=5)


search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": None,
}

tools = Toolset()
tools.add(search_tool, search)

thread = Thread(SearchAgentLLMInstruction())
question = "I just discovered the course. Can I join it?"
thread.add_user_message(question)

prompter = LLMPrompter()

agent = Agent(prompter, thread)

response_text = agent.invoke(tools)

print(response_text)
