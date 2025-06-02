import json
import os
from dotenv import load_dotenv
from qwen_agent.llm import get_chat_model
import yaml
import time

from smolagents import (
    LiteLLMModel, CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool
)
from tools.final_answer import FinalAnswerTool

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


def call_the_agent(query: str, url: str = None):
    """
    Call an agent to search the web for updated information or access a webpage url.
    Args:
        query: The query to search for, in str
        url: The URL to visit, if applicable
    Returns:
        the response from the agent, in str
    """
    model_ollama = LiteLLMModel(
        model_id="ollama_chat/" + os.getenv("OLLAMA_AGENT_MODEL"),
        api_base=os.getenv("OLLAMA_AGENT_ENDPOINT"),
        temperature=0.1,
        num_ctx=int(os.getenv("OLLAMA_AGENT_MAX_TOKENS")),
    )

    # final_answer = FinalAnswerTool()

    # with open("prompts.yaml", 'r') as stream:
    #     prompt_templates = yaml.safe_load(stream)

    custom_agent = CodeAgent(
        name=os.getenv("OLLAMA_AGENT_NAME"),
        description=os.getenv("OLLAMA_AGENT_DESCRIPTION"),
        model=model_ollama,
        tools=[DuckDuckGoSearchTool()],
        add_base_tools=False,
        max_steps=10,
        verbosity_level=0,
        # prompt_templates=prompt_templates,
        additional_authorized_imports=[
            "requests", "bs4", "datetime", "matplotlib.pyplot", "pytz", "csv", "yaml", "io", "os",
            "posixpath", "zlib", "json", "pandas",
        ],
    )

    final_agent_output = custom_agent.run(
        query
    )

    return str(final_agent_output)


def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }


def get_function_by_name(name):
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date
    if name == "call_the_agent":
        return call_the_agent


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_the_agent",
            "description": "Call an agent to perform complex tasks such as searching the web for updated information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for, in str",
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL to visit, if applicable",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

MESSAGES = [
    {"role": "system", "content": os.getenv("OLLAMA_MODEL_SYSTEM_PROMPT")},
    {"role": "user",  "content": user_input},
]

messages = MESSAGES[:]

functions = [tool["function"] for tool in TOOLS]

last_content = ""

for chunk in llm.chat(messages=messages, functions=functions, stream=True):
    if isinstance(chunk, list) and len(chunk) > 0:
        message = chunk[0]
        if message.get("role") == "assistant":
            content = message.get("content", "")
            delta = content[len(last_content):]
            print(delta, end="", flush=True)
            time.sleep(0.01)
            last_content = content

messages.extend(chunk)

if messages[-1].get('function_call'):
    print("\n\nChamando funcao " + str(messages[-1].get('function_call')['name']) + "\n\n")

    for message in chunk:
        if fn_call := message.get("function_call", None):
            fn_name: str = fn_call['name']
            fn_args: dict = json.loads(fn_call["arguments"])
            fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args), ensure_ascii=False)
            messages.append({
                "role": "function",
                "name": fn_name,
                "content": fn_res,

            })

    for responses in llm.chat(messages=messages, functions=functions):
        pass
    messages.extend(responses)

    print(messages[-1]["content"])
