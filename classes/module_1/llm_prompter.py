import os

import dotenv
import openai

from classes.module_1.thread import Thread


dotenv.load_dotenv()


class LLMPrompter():
    def __init__(self) -> None:
        self._client = openai.Client(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY")
        )

    def generate(self, thread: Thread, model: str ="nemotron-3-ultra:cloud") -> str:
        response = self._client.responses.create(
            model=model,
            input=thread.get_messages(),
        )
        return response.output_text
