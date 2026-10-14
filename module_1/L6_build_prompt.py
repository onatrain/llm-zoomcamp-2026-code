from L5_minsearch import question, search_results

from utils.build_prompt import build_prompt


def build_context(documents: list[dict[str, str]]) -> str:
    """Format search results into a context block for the prompt.

    Args:
        documents: The search result documents.

    Returns:
        A formatted context string.
    """
    lines = []

    for doc in documents:
        lines.append(doc["section"])
        lines.append(f"Q: {doc['question']}")
        lines.append(f"A: {doc['answer']}")
        lines.append("")

    return "\n".join(lines).strip()


context = build_context(search_results)

prompt = build_prompt(question, context)

print(prompt)
