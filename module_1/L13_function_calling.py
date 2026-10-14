import json
import os
from pathlib import Path
from typing import cast

import dotenv
from openai import Client
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from classes.local_faq import LocalFAQ

current_dir = Path(__file__).parent
index_path = str(current_dir / "data/faq.db")

faq = LocalFAQ(index_path)
faq.open()

dotenv.load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# Tool definition
# ═══════════════════════════════════════════════════════════════════
def search(query: str) -> list[dict[str, str]]:
    """Search the FAQ database with boosted and filtered fields.

    Args:
        query: The search query text.

    Returns:
        A list of matching FAQ entries as dictionaries.
    """
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return faq.search(query, filter_dict=filter_dict, boost_dict=boost_dict, num_results=5)


# ═══════════════════════════════════════════════════════════════════
# Tool definition structure
# ═══════════════════════════════════════════════════════════════════
search_tool: FunctionToolParam = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": None,
}

client = Client(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

model = "nemotron-3-ultra:cloud"

# ═══════════════════════════════════════════════════════════════════
# List of messages for the model
# ═══════════════════════════════════════════════════════════════════
messages: list[ResponseInputItemParam] = [
    EasyInputMessageParam(role="user", content="I just discovered the course. Can I join it?")
]

response = client.responses.create(
    model=model,
    input=messages,
    tools=[search_tool],
)

# ═══════════════════════════════════════════════════════════════════
# Correctly get the tool call. The Responses API does not fix an order
# in the objects it uses to build the response.
# ═══════════════════════════════════════════════════════════════════
tool_call: ResponseFunctionToolCall

for output in response.output:
    if isinstance(output, ResponseFunctionToolCall):
        tool_call = output
        break

# ═══════════════════════════════════════════════════════════════════
# Get the arguments with which the tool must be called
# ═══════════════════════════════════════════════════════════════════
args = json.loads(tool_call.arguments)

# ═══════════════════════════════════════════════════════════════════
# Call the tool and serialize the result into a JSON
# ═══════════════════════════════════════════════════════════════════
results = search(**args)
result_json = json.dumps(results, indent=2)

# ═══════════════════════════════════════════════════════════════════════
# Store the tool call made by the model
# ═══════════════════════════════════════════════════════════════════════
messages.append(cast(ResponseInputItemParam, tool_call))

# ═══════════════════════════════════════════════════════════════════
# Store the tool call response
# ═══════════════════════════════════════════════════════════════════
function_output: FunctionCallOutput = {
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": result_json,
}
messages.append(function_output)

# ═══════════════════════════════════════════════════════════════════
# Second call supplying the tool call response
# ═══════════════════════════════════════════════════════════════════
response = client.responses.create(
    model=model,
    input=messages,
    tools=[search_tool],
)

print(response.output_text)

# ═══════════════════════════════════════════════════════════════════
# Proper handling of response.usage
# ═══════════════════════════════════════════════════════════════════
if response.usage is not None:
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
else:
    print("No usage data available")

"""
[
    {
        'role': 'user',
        'content': 'I just discovered the course. Can I join it?', 
    },
    ResponseFunctionToolCall(
        arguments='{"query":"join course enrollment register"}', 
        call_id='call_ymnqsiq5', 
        name='search', 
        type='function_call', 
        id='fc_resp_166709_0', 
        namespace=None, 
        status='completed'
    ),
    {
        'call_id': 'call_ymnqsiq5',
        'output': '[\n'
                '  {\n'
                '    "id": "74eb249bbf",\n'
                '    "course": "llm-zoomcamp",\n'
                '    "section": "General Course-Related Questions",\n'
                '    "question": "I just discovered the course. Can I still '
                'join?",\n'
                '    "answer": "Yes, but if you want to receive a certificate, you '
                'need to submit your project while we\\u2019re still accepting '
                'submissions."\n'
                '  }\n'
                ']',
        'type': 'function_call_output'
    }
]
"""
