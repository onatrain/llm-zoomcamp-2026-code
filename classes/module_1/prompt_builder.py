from utils.build_prompt import build_prompt


class PromptBuilder():
    def build(self, question: str, retrieval: list[dict[str, str]]) -> str:
        context = self._build_context(retrieval)

        prompt = build_prompt(question, context)

        return prompt

    def _build_context(self, documents: list[dict[str, str]]) -> str:
        lines = []

        for doc in documents:
            lines.append(doc["section"])
            lines.append(f"Q: {doc['question']}")
            lines.append(f"A: {doc['answer']}")
            lines.append("")

        return "\n".join(lines).strip()

