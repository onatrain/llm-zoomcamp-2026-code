import os

import dotenv
import openai

dotenv.load_dotenv()

client = openai.Client(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

if __name__ == "__main__":
    print(client.models.list())