"""Util that calls bm25."""
from typing import Any, Dict, List, Optional
import os
import requests
from rank_bm25 import BM25Okapi

MAX_QUERY_LENGTH = 300

BM25_description = """The BM25 Search tool provides access to a vast collection of articles covering a wide range of topics.
Can query specific keywords or topics to retrieve accurate and comprehensive information.
"""


class BM25API:

    def query(self, text: str, candidates: list, ):

        tokenized_corpus = [doc.split(" ") for doc in candidates]

        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = text.split(" ")

        doc_scores = bm25.get_scores(tokenized_query)

        return doc_scores


class BM25Run():
    """Tool that adds the capability to search using the Wikipedia API."""

    name = "wikipedia_search"
    signature = f"{name}(query: str) -> str"
    description = BM25_description
    api_wrapper = BM25API()
    max_results = 5

    def __init__(self, max_results):
        self.max_results = max_results

    def flatten_corpus(self, corpus):
        candidates = []
        for obj in corpus:
            candidates.append(f"Title: {obj['title']}\nText: {obj['paragraph_text']}")

        return candidates

    def __call__(
        self,
        query: str,
        corpus: list
    ) -> str:
        """Use the DDGS tool."""

        max_try = -1 # used when the query is bad, and we do not want to retry so many time

        candidates = self.flatten_corpus(corpus)

        while True:

            try:
                output = [r for r in self.api_wrapper.query(query[:MAX_QUERY_LENGTH], candidates,)]
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

                print(f"try again. BM25 raise the error: {E}")

        evidences = []

        top_indices = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:self.max_results]

        for index in top_indices:
            evidences.append({
                "title": corpus[index]["title"],
                "text": corpus[index]["paragraph_text"]
            })

        if len(evidences) == 0:
            # do not return anything from search engine, add dummy
            evidences.append({
                "title": "dummy",
                "text": "the search engine did not return anything"
            })

        return evidences, top_indices

if __name__ == '__main__':

    engine = BM25Run(max_results=5)
    corpus = [
        {"title": "1", "text": "Hello there good man!"},
        {"title": "2", "text": "It is quite windy in London"},
        {"title": "3", "text": "How is the weather today?"},
    ]
    print(engine("What is the weather today in Beijing?", corpus))
