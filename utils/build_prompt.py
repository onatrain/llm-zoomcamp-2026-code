LLM_INSTRUCTION = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, context=context)
