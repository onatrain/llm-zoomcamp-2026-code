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

from module_1.classes.local_faq import LocalFAQ

current_dir = Path(__file__).parent
index_path = str(current_dir / "data/faq.db")

faq = LocalFAQ(index_path)
faq.open()

dotenv.load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# Definición del tool
# ═══════════════════════════════════════════════════════════════════
def search(query: str) -> list[dict[str, str]]:
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return faq.search(query, filter_dict=filter_dict, boost_dict=boost_dict, num_results=5)


# ═══════════════════════════════════════════════════════════════════
# Estructura de definición del tool
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
# Lista de mensajes para el modelo
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
# Obtener correctamente la tool call. El Responses API no fija un orden
# en los objetos con los que construye la respuesta.
# ═══════════════════════════════════════════════════════════════════
tool_call: ResponseFunctionToolCall

for output in response.output:
    if isinstance(output, ResponseFunctionToolCall):
        tool_call = output
        break

# ═══════════════════════════════════════════════════════════════════
# Obtiene los argumentos con los que se debe llamar el tool
# ═══════════════════════════════════════════════════════════════════
args = json.loads(tool_call.arguments)

# ═══════════════════════════════════════════════════════════════════
# Se llama al tool y se serializa el resultado en un json
# ═══════════════════════════════════════════════════════════════════
results = search(**args)
result_json = json.dumps(results, indent=2)

# ═══════════════════════════════════════════════════════════════════════
# Se guarda el tool call que hizo el modelo
# ═══════════════════════════════════════════════════════════════════════
messages.append(cast(ResponseInputItemParam, tool_call))

# ═══════════════════════════════════════════════════════════════════
# Se guarda la respuesta del tool call
# ═══════════════════════════════════════════════════════════════════
function_output: FunctionCallOutput = {
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": result_json,
}
messages.append(function_output)

# ═══════════════════════════════════════════════════════════════════
# Segunda llamada suministrando el tool call response
# ═══════════════════════════════════════════════════════════════════
response = client.responses.create(
    model=model,
    input=messages,
    tools=[search_tool],
)

print(response.output_text)

# ═══════════════════════════════════════════════════════════════════
# Manejo correcto de response.usage
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
