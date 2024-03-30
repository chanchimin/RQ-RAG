import json
import os
from pprint import pprint
import requests
from typing import Any, Dict, List, Optional
import os
import requests

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY'] = "your key"
os.environ['BING_SEARCH_V7_ENDPOINT'] = "https://api.bing.microsoft.com"
subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
endpoint = os.environ['BING_SEARCH_V7_ENDPOINT'] + "/v7.0/search"
MAX_QUERY_LENGTH = 300

BingSearch_DESCRIPTION = """The bing search tool provides access to a vast collection of articles covering a wide range of topics.
Can query specific keywords or topics to retrieve accurate and comprehensive information.
"""

class BingSearchAPI:

    def query(self, text: str):

        params = {'q': text, 'mkt': 'en-US'}
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}

        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        if response.status_code == 200:
            if len(response.json()['webPages']['value']) == 0:
                # do not know why, retry
                raise ValueError("status==200, but do not know why return nothing, retry")
            return response.json()['webPages']['value']

        else:
            raise ValueError(f" bing search error\n{response}")


class BingSearchQueryRun():
    """Tool that adds the capability to search using the Wikipedia API."""

    name = "bing_search"
    signature = f"{name}(query: str) -> str"
    description = BingSearch_DESCRIPTION
    api_wrapper = BingSearchAPI()

    def __init__(self, max_results):
        self.max_results = max_results

    def __call__(
        self,
        query: str,
    ) -> str:
        """Use the DDGS tool."""

        max_try = -1 # used when the query is bad, and we do not want to retry so many time

        while True:

            try:
                output = [r for r in self.api_wrapper.query(query[:MAX_QUERY_LENGTH])]
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

                print(f"try again. bing raise the error: {E}")

        evidences = []

        for ins in output:
            evidences.append({
                "title": ins["name"],
                "text": ins["snippet"]
            })

        if len(evidences) == 0:
            # do not return anything from search engine, add dummy
            evidences.append({
                "title": "dummy",
                "text": "the search engine did not return anything"
            })

        return evidences[:self.max_results]

if __name__ == '__main__':

    engine = BingSearchQueryRun(max_results=5)
    print(engine("What is the weather today in Beijing?"))
