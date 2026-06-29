import json
import os
from dataclasses import dataclass
from typing import Any

import dotenv
from openai import Client
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)
from openai.types.responses import ResponseOutputItem

from module_1.classes.thread import Thread

dotenv.load_dotenv()


@dataclass
class MessageResponse():
    text: str
    type: str = "message"


@dataclass
class FunctionCallResponse():
    call_id: str
    name: str
    arguments: dict[str, Any]
    type: str = "function_call"


@dataclass
class AgentResponse():
    id: str
    data: list[MessageResponse | FunctionCallResponse]


class LLMPrompter():
    def __init__(self) -> None:
        self._client = Client(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY")
        )

    def generate(self, thread: Thread, model: str ="nemotron-3-ultra:cloud") -> str:
        response = self._client.responses.create(
            model=model,
            input=thread.get_input(),
            instructions=thread.get_instruction(),
            store=False,
        )
        return response.output_text

    def agentic_call(self, thread: Thread, tools: list[FunctionToolParam], model: str ="nemotron-3-ultra:cloud") -> AgentResponse:
        response = self._client.responses.create(
            model=model,
            input=thread.get_input(),
            tools=tools,
            store=False,
        )

        return self._process_output(response.id, response.output)

    def _process_output(self, id: str, output: list[ResponseOutputItem]) -> AgentResponse:
        return_items: list[MessageResponse | FunctionCallResponse] = []

        for item in output:
            match item:
                case i if type(i) is ResponseOutputMessage:
                    return_items.append(self._get_message_response(item))
                case i if type(i) is ResponseFunctionToolCall:
                    return_items.append(self._get_call_response(item))
                case _:
                    pass

        return AgentResponse(id, return_items)

    def _get_message_response(self, item: ResponseOutputMessage) -> MessageResponse:
        text_chunks = [c.text for c in item.content if c.type == "output_text"]
        return MessageResponse(text="".join(text_chunks))

    def _get_call_response(self, item: ResponseFunctionToolCall) -> FunctionCallResponse:
        return FunctionCallResponse(
            call_id=item.call_id,
            name=item.name,
            arguments=json.loads(item.arguments)
        )
