import json
import os
from dotenv import load_dotenv
from qwen_agent.llm import get_chat_model

load_dotenv('./.env')

llm = get_chat_model({
    "model": os.getenv("OLLAMA_MODEL"),
    "model_server": os.getenv("OLLAMA_ENDPOINT"),
    "api_key": os.getenv("OLLAMA_KEY"),
})


def today_passkey():
    """
    Get today's passkey.
    Returns:
        the passkey for today, in str
    """
    return "blablabla-102030"


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
    if name == "today_passkey":
        return today_passkey


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
            "name": "today_passkey",
            "description": "Get today's passkey.",
        },
    },
]

MESSAGES = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user",  "content": input("User: ")},
]

messages = MESSAGES[:]

functions = [tool["function"] for tool in TOOLS]

for responses in llm.chat(
    messages=messages,
    functions=functions,
    extra_generate_cfg=dict(parallel_function_calls=True),
):
    pass
messages.extend(responses)

for message in responses:

    if fn_call := message.get("function_call", None):
        fn_name: str = fn_call['name']
        fn_args: dict = json.loads(fn_call["arguments"])
        fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))
        messages.append({
            "role": "function",
            "name": fn_name,
            "content": fn_res,

        })

for responses in llm.chat(messages=messages, functions=functions):
    pass
messages.extend(responses)

print(messages)
