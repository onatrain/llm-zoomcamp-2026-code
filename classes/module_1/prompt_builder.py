from classes.module_1.thread import Thread
from utils.build_prompt import build_prompt


class PromptBuilder():
    def __init__(self, thread: Thread):
        self._thread = thread

    def build(self, question: str, retrieval: list[dict[str, str]]) -> Thread:
        context = self._build_context(retrieval)
        user_prompt = build_prompt(question, context)

        self._thread.add_message("user", user_prompt)

        return self._thread

    def _build_context(self, documents: list[dict[str, str]]) -> str:
        lines = []

        for doc in documents:
            lines.append(doc["section"])
            lines.append(f"Q: {doc['question']}")
            lines.append(f"A: {doc['answer']}")
            lines.append("")

        return "\n".join(lines).strip()

