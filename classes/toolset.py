from dataclasses import dataclass
from typing import Any, Callable, cast

from openai.types.responses import FunctionToolParam


@dataclass
class AgentTool:
    definition: FunctionToolParam
    callback: Callable[[Any], Any]


class Toolset():
    def __init__(self) -> None:
        self._tools: list[AgentTool] = []

    def add(self, definition: dict[str, Any], callback: Callable[[Any], Any]) -> None:
        new_tool = AgentTool(definition=cast(FunctionToolParam, definition), callback=callback)
        self._tools.append(new_tool)

    def get_definitions(self) -> list[FunctionToolParam]:
        return [t.definition for t in self._tools]

    def get_callbacks(self) -> dict[str, Callable[[Any], Any]]:
        return {t.definition["name"]: t.callback for t in self._tools}
