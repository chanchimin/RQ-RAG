import os
import requests
import json
from openai import OpenAI


class OpenAIClient:
    def __init__(self, config, model_version):

        self.model_version = model_version
        self.client = OpenAI(**config)



    def chat_sync(self, messages, params={"temperature": 0, "max_tokens": 500}):

        return self.client.chat.completions.create(
                model=self.model_version,
                messages=messages,
                **params
            )

