import json
import dotenv, os

from pathlib import Path

from openai import Client

from classes.module_1.local_faq import LocalFAQ


current_dir = Path(__file__).parent

index_path = str(current_dir / "data/faq.db")


def search(query):
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}
    faq = LocalFAQ(index_path)

    faq.open()

    return faq.search(query, filter_dict=filter_dict, boost_dict=boost_dict, num_results=5)


dotenv.load_dotenv()

search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

client = Client(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

model="nemotron-3-ultra:cloud"
messages = [
    {"role": "user", "content": "I just discovered the course. Can I join it?"}
]

response = client.responses.create(
    model=model,
    input=messages,
    tools=[search_tool],
)

call = response.output[-1]
args = json.loads(call.arguments)

results = search(**args)
result_json = json.dumps(results, indent=2)

messages.extend(response.output)

messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": result_json,
})

response = client.responses.create(
    model=model,
    input=messages,
    tools=[search_tool],
)

print(response.output_text)

usage = response.usage
print("\n", "Usage:")
print("Input tokens:", usage.input_tokens)
print("Output tokens:", usage.output_tokens)

"""
[
    ResponseReasoningItem(
        id='rs_resp_362423', 
        summary=[
            Summary(
                text='', 
                type='summary_text'
            )
        ], 
        type='reasoning', 
        content=None, 
        encrypted_content='', 
        status=None
    ), 
    ResponseFunctionToolCall(
        arguments='{"query":"join course enrollment registration"}', 
        call_id='call_function_avbm0zm9g7hw_1', 
        name='search', 
        type='function_call', 
        id='fc_resp_362423_0', 
        namespace=None, 
        status='completed'
    )
]
"""