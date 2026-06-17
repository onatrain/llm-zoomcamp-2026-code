import os

import dotenv
import openai
from openai.types.responses import ResponseInputParam

dotenv.load_dotenv()

client = openai.Client(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

def llm(user_prompt: str, system_prompt: str | None = None):
    prompt_history: ResponseInputParam = []
    if system_prompt:
        prompt_history.append({"role": "system", "content": system_prompt})
    prompt_history.append({"role": "user", "content": user_prompt})

    response = client.responses.create(
        model="nemotron-3-ultra:cloud",
        input=prompt_history,
    )
    return response.output_text
