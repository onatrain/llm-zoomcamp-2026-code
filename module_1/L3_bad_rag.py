from utils.llm import llm


def search(question: str) -> str:
    context = {
        "I just discovered the course. Can I still join?": (
            "Yes, but if you want to receive a certificate, you need to submit your project "
            "while we're still accepting submissions."
        ),
        "I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?": (
            "You don't need it. You're accepted. You can also just start learning and submitting homework "
            "(while the form is open) without registering. It is not checked against any registered list. "
            "Registration is just to gauge interest before the start date."
        ),
        "What is the video/zoom link to the stream for the \"Office Hours\" or live/workshop sessions?": (
            "The zoom link is only published to instructors/presenters/TAs. Students participate via YouTube "
            "Live and submit questions to Slido."
        ),
        "Cloud alternatives with GPU": (
            "Check the quota and reset cycle carefully. Potential options include Google Colab, Kaggle, Databricks."
        ),
    }
    return context.get(question, "")


def build_prompt(question: str, search_results: str) -> str:
    prompt = f"""
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."

Question:
{question}

Context:
{search_results}
"""
    return prompt


def rag(question: str) -> str:
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)


question = "I just discovered the course. Can I still join?"

print(rag(question))