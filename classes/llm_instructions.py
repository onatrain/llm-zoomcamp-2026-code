from dataclasses import dataclass, field
from typing import Protocol

from openai.types.responses import EasyInputMessageParam

PIPELINE_INSTRUCTION = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
""".strip()

SEARCH_AGENT_INSTRUCTION = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()


class LLMInstruction(Protocol):
    prompt: EasyInputMessageParam
    message: str


@dataclass
class PipelineLLMInstruction:
    prompt: EasyInputMessageParam = field(
        default_factory=lambda: EasyInputMessageParam(role="system", content=PIPELINE_INSTRUCTION)
    )
    message: str = PIPELINE_INSTRUCTION


@dataclass
class SearchAgentLLMInstruction:
    prompt: EasyInputMessageParam = field(
        default_factory=lambda: EasyInputMessageParam(role="system", content=SEARCH_AGENT_INSTRUCTION)
    )
    message: str = SEARCH_AGENT_INSTRUCTION

