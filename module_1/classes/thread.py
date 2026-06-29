import json
from typing import Any, Literal

from openai.types.responses import EasyInputMessageParam, ResponseInputItemParam, ResponseFunctionToolCallParam
from openai.types.responses.response_input_item_param import FunctionCallOutput

from module_1.classes.llm_instructions import LLMInstruction

role_type = Literal["user", "assistant"]


class Thread:
    def __init__(self, llm_instruction: LLMInstruction):
        self._items: list[ResponseInputItemParam] = []
        self._instruction = llm_instruction.message

    def add_user_message(self, content: str) -> None:
        """Adds a user message to the thread."""
        self._items.append(EasyInputMessageParam(
            type="message",
            role="user",
            content=content,
        ))

    def add_assistant_text(self, content: str) -> None:
        """Adds an assistant message to the thread (without tool calls)."""
        self._items.append(EasyInputMessageParam(
            type="message",
            role="assistant",
            content=content,
        ))

    def add_function_call(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        """Adds a function call to the thread."""
        self._items.append(ResponseFunctionToolCallParam(
            type="function_call",
            call_id=call_id,
            name=name,
            arguments=json.dumps(arguments),
        ))

    def add_function_call_output(self, call_id: str, output: Any) -> None:
        """Adds a results function call message to the thread."""
        self._items.append(FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=json.dumps(output) if isinstance(output, (dict, list)) else str(output),
        ))

    def get_input(self) -> list[ResponseInputItemParam]:
        """Return the full items list."""
        return self._items.copy()

    def get_instruction(self) -> str:
        """Return the system instruction."""
        return self._instruction

    def __repr__(self) -> str:
        return repr(self._items)
