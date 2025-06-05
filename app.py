from tools.qwen_functions import QwenFunctions
from qwen_agent.llm import get_chat_model
from dotenv import load_dotenv
import os
import json
import time

load_dotenv('./.env')

user_input = input("User input: ")

llm_cfg = {
    "model": os.getenv("OLLAMA_MODEL"),
    "model_server": os.getenv("OLLAMA_MODEL_ENDPOINT") + "/v1",
    "api_key": os.getenv("OLLAMA_KEY"),
    "generate_cfg": {
        "max_tokens": int(os.getenv("OLLAMA_MODEL_MAX_TOKENS", 2048)),
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }
}

llm = get_chat_model(llm_cfg)


MESSAGES = [
    {"role": "system", "content": os.getenv("OLLAMA_MODEL_SYSTEM_PROMPT")},
    {"role": "user",  "content": user_input},
]

messages = MESSAGES[:]

functions = [tools["function"] for tools in QwenFunctions.custom_tools()]

last_content = ""

for chunk in llm.chat(messages=messages, functions=functions, stream=True):
    if isinstance(chunk, list) and len(chunk) > 0:
        message = chunk[0]
        if message.get("role") == "assistant":
            content = message.get("content", "")
            delta = content[len(last_content):]
            print(delta, end="", flush=True)
            last_content = content

messages.extend(chunk)

if messages[-1].get('function_call'):
    print("\n\nChamando funcao " + str(messages[-1].get('function_call')['name']) + "\n\n")

    for message in chunk:
        if fn_call := message.get("function_call", None):
            fn_name: str = fn_call['name']
            fn_args: dict = json.loads(fn_call["arguments"])
            fn_res: str = json.dumps(QwenFunctions.get_function_by_name(fn_name)(**fn_args), ensure_ascii=False)
            messages.append({
                "role": "function",
                "name": fn_name,
                "content": fn_res,

            })

    for responses in llm.chat(messages=messages, functions=functions):
        pass
    messages.extend(responses)

    print(messages[-1]["content"])
