"""Util that calls DockDockGo."""
from typing import Any, Dict, List, Optional
import os
import requests
from duckduckgo_search import DDGS
import time

MAX_QUERY_LENGTH = 300

DockDockGo_DESCRIPTION = """The Wikipedia Search tool provides access to a vast collection of articles covering a wide range of topics.
Can query specific keywords or topics to retrieve accurate and comprehensive information.
"""

ddgs = DDGS(timeout=20)

class RapidAPI:

    def __init__(self, rapidapi_name):
        self.rapidapi_name = rapidapi_name

    def query(self, text: str, max_results: int):

        time.sleep(1)
        response = ddgs.text(text.strip("'"), max_results=10)
        return response[:max_results]

class DDGSQueryRun():
    """Tool that adds the capability to search using the Wikipedia API."""

    name = "wikipedia_search"
    signature = f"{name}(query: str) -> str"
    description = DockDockGo_DESCRIPTION
    max_results = 5

    def __init__(self, max_results, rapidapi_name="one"):
        self.max_results = max_results
        self.api_wrapper = RapidAPI(rapidapi_name)

    def __call__(
        self,
        query: str,
    ) -> str:
        """Use the DDGS tool."""

        max_try = -1 # used when the query is bad, and we do not want to retry so many time

        while True:

            try:
                output = [r for r in self.api_wrapper.query(query[:MAX_QUERY_LENGTH], max_results=self.max_results)]
                break
            except Exception as E:

                if isinstance(E, ValueError):

                    # the first time encounter bad query
                    if max_try == -1:
                        max_try = 5

                    if max_try == 0:
                        output = []
                        break

                    # cur wrong text "What ... ?|Which ...?|"
                    query = query.split("|")[0]
                    max_try -= 1

                print(f"try again. dockdockgo raise the error: {E}")

        evidences = []

        for ins in output:
            evidences.append({
                "title": ins["title"],
                "text": ins["description"] if "description" in ins else ins["body"]
            })

        if len(evidences) == 0:
            # do not return anything from search engine, add dummy
            evidences.append({
                "title": "dummy",
                "text": "the search engine did not return anything"
            })

        return evidences

if __name__ == '__main__':

    engine = DDGSQueryRun(max_results=5)
    print(engine("What is the weather today in Beijing?"))
