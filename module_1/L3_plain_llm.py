from L2_connect import client


def llm(prompt):
    response = client.responses.create(
        model="nemotron-3-ultra:cloud",
        input=prompt
    )
    return response.output_text


question = "I just discovered the course. Can I join now?"
answer = llm(question)

print(answer)