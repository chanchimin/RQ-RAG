import os
import faiss
from openai import OpenAI
import torch
import numpy as np
import sys
sys.path.append("../..")
import retrieval_lm.src.normalize_text as normalize_text


class OpenAIEmbedSearch:
    def __init__(self, ndocs, task, args, use_calculated_embeds=True, is_train=False):
        self.ndocs = ndocs
        self.task = task
        self.args = args
        self.client = OpenAI()

        # set the question embedding save path and retrieval results save path
        if use_calculated_embeds:

            if is_train:

                embeddings_path = os.path.join(os.path.dirname(self.args.input_file), "train_context_embeddings.pt")
                self.all_embeddings = torch.load(embeddings_path)

            else:

                embeddings_path = os.path.join(os.path.dirname(self.args.input_file), "test_context_embeddings.pt")
                self.all_embeddings = torch.load(embeddings_path)

    def __call__(self, query: str, corpus: list, index: int = None):

        # get the normalized text and embedding

        normalized_query = query.lower()
        normalized_query = normalize_text.normalize(normalized_query)

        try:
            normalized_query_emb = self.client.embeddings.create(input=[normalized_query], model="text-embedding-3-large").data[0].embedding

        except Exception as E:
            print(f"bad request for openai, use dummy input : {E}")
            normalized_query_emb = self.client.embeddings.create(input=["dummy"], model="text-embedding-3-large").data[0].embedding

        normalized_query_emb = torch.tensor(normalized_query_emb)

        top_indices = self.find_most_similar_context(normalized_query_emb, index)

        evidences = []
        for top_index in top_indices:
            if "title" in corpus[top_index]:
                evidences.append({
                    "title": corpus[top_index]["title"],
                    "text": corpus[top_index]["paragraph_text"]
                })
            elif "text" in corpus[top_index]:
                evidences.append({
                    "title": "Retrieved Documents for Reference",
                    "text": corpus[top_index]["text"].split("*****")[-1]
                })

        if len(evidences) == 0:
            # do not return anything from search engine, add dummy
            evidences.append({
                "title": "dummy",
                "text": "the search engine did not return anything"
            })

        return evidences, top_indices

    def cosine_similarity(self, a, b):
        """calc the similarity of a and b"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def find_most_similar_context(self, query_embedding, cur_index):
        """find the most relevant context

        parameter:
        query_embedding -- query embedding (1D numpy array)
        context_embeddings -- context list (2D numpy array)

        return:
        the most similar index and embedding
        """

        if cur_index is not None:
            cur_context_embeddings = self.all_embeddings[cur_index]
        else:
            cur_context_embeddings = self.all_embeddings

        similarities = [self.cosine_similarity(query_embedding, context_embedding) for context_embedding in
                        cur_context_embeddings]

        most_similar_index = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:self.ndocs]

        return most_similar_index
