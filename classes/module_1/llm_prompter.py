import os

import dotenv
import openai

dotenv.load_dotenv()


class LLMPrompter():
    def __init__(self) -> None:
        self._client = openai.Client(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY")
        )

    def generate(self, prompt: str, model="nemotron-3-ultra:cloud") -> str:
        response = self._client.responses.create(
            model=model,
            input=prompt
        )
        return response.output_text
