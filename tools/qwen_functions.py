from .call_agent import CallAgent


class QwenFunctions():
    def get_function_by_name(name):
        if name == "get_current_temperature":
            return QwenFunctions.get_current_temperature
        if name == "get_temperature_date":
            return QwenFunctions.get_temperature_date
        if name == "call_agent":
            return CallAgent.call_smolagent

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

    def custom_tools():
        return [
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
                    "name": "call_agent",
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
