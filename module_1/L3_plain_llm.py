from L2_connect import client


def llm(prompt: str) -> str:
    """Send a prompt to the LLM and return its text response.

    Args:
        prompt: The prompt text to send.

    Returns:
        The model's output text.
    """
    response = client.responses.create(
        model="nemotron-3-ultra:cloud",
        input=prompt
    )
    return str(response.output_text)


question = "I just discovered the course. Can I join now?"
answer = llm(question)

print(answer)