from .base import LMAgent
import openai
import logging
import traceback
from .datatypes import Action
import backoff
from .prompt_template.Template import QueryJudgerTemplate
from string import Template
import re
from .openai_api import OpenAIClient

LOGGER = logging.getLogger("Root")


class OpenAIJudgerLMAgent(LMAgent):
    def __init__(self, api_type, config, model_version="gpt-3.5-turbo"):
        super().__init__(config)
        assert api_type in ["azure", "openai"]

        if api_type == "openai":
            self.api = OpenAIClient(config, model_version)

        self.usage_profiles = []
        self.max_try = 3

    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.APIError,
            openai.Timeout,
            openai.RateLimitError,
            openai.APIConnectionError,
        ),
    )
    def call_lm(self, messages):

        # Prepend the prompt with the system message
        js = self.api.chat_sync(messages=messages)

        response = js.choices[0].message.content

        # usage contains input token, output token, times
        usage = {
            "promptTokens": js.usage.prompt_tokens,
            "completionTokens": js.usage.completion_tokens,
            "totalTokens": js.usage.total_tokens,
            "costTimeMillis": 0
        }

        final_response = response.replace('"', "'")
        self.usage_profiles.append(usage)

        return final_response, usage

    def act(self, template: str, **kwargs):

        # TODO construct messages here

        llm_query = Template(template).substitute(kwargs)

        messages = [{
            "role": "user",  # user / assistant
            "content": f"{llm_query}",
        }]

        for _ in range(self.max_try):

            try:
                lm_output, usage = self.call_lm(messages)
                return lm_output
            except KeyboardInterrupt:
                exit()
            except Exception as E:  # mostly due to model context window limit
                tb = traceback.format_exc()
                print(f"Some error happens when calling judger agent: \n{tb}")

        return f"InvalidRequestError"

