import requests
import os
from dotenv import load_dotenv
from smolagents import (
    LiteLLMModel, CodeAgent, DuckDuckGoSearchTool, tool
)

load_dotenv('./.env')


class CallAgent():
    """
    A class to call the smolagent to process complex user queries.
    """
    def call_smolagent(query: str, url: str = None):
        """
        Call the smolagent to process complex user queries
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

        @tool
        def visit_webpage_md(url: str) -> str:
            """A tool that fetches the content of a webpage and returns it as markdown.
            Args:
                url: A string representing the URL of the webpage.
            """
            try:
                response = requests.get('https://r.jina.ai/' + url)
                response.raise_for_status()

                markdown_content = response.text.strip()
                return markdown_content

            except Exception as e:
                return f"An unexpected error occurred: {str(e)}"

        custom_agent = CodeAgent(
            name=os.getenv("OLLAMA_AGENT_NAME"),
            description=os.getenv("OLLAMA_AGENT_DESCRIPTION"),
            model=model_ollama,
            tools=[DuckDuckGoSearchTool(), visit_webpage_md],
            add_base_tools=False,
            max_steps=10,
            verbosity_level=2,
            additional_authorized_imports=[
                "requests", "bs4", "datetime", "matplotlib.pyplot", "pytz", "csv", "yaml",
                "io", "os", "posixpath", "zlib", "json", "pandas",
            ],
        )

        final_anwer = custom_agent.run(
            query
        )

        return str(final_anwer)
