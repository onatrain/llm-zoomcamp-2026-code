from typing import Literal, Any
from classes.module_1.llm_instructions import LLMInstruction


role_type = Literal["user", "assistant"]


class Thread:
    def __init__(self, llm_instruction: LLMInstruction):
        self._thread = [llm_instruction.message]

    def add_message(self, role: role_type, content: str) -> None:
        message = {"role": role, "content": content}
        self._thread.append(message)

    def add_tool_response(self) -> None:
        pass

    def add_raw_response(self) -> None:
        pass

    def get_messages(self) -> list[Any]:
        return self._thread
