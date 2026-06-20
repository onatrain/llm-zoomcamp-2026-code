from dataclasses import dataclass, field
from typing import Protocol


PIPELINE_INSTRUCTION = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
""".strip()


class LLMInstruction(Protocol):
    message: dict[str, str]

@dataclass
class PipelineLLMInstruction:
    message: dict[str, str] = field(default_factory=lambda: {"role": "system", "content": PIPELINE_INSTRUCTION})
