import jsonlines
import os
import src.normalize_text
import argparse
import csv
import logging
import pickle
import numpy as np
import torch
import transformers
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

def embed_passages(args, input_data, context_num):

    all_embeddings = []

    for ins_num, ins in tqdm(enumerate(input_data), total=len(input_data)):
        context_embeddings_for_cur_ins = []

        for context in ins["contexts"]:
            text = context["title"] + " " + context["paragraph_text"]
            text = text.lower()
            text = src.normalize_text.normalize(text)
            text_embedding = client.embeddings.create(input=[text], model="text-embedding-3-large").data[0].embedding
            context_embeddings_for_cur_ins.append(text_embedding)

        if len(context_embeddings_for_cur_ins) < context_num:
            while len(context_embeddings_for_cur_ins) < context_num:
                context_embeddings_for_cur_ins.append([-100] * len(context_embeddings_for_cur_ins[0]))

        all_embeddings.append(context_embeddings_for_cur_ins)

    all_embeddings = torch.tensor(all_embeddings)

    torch.save(all_embeddings, args.output_file)
    assert all_embeddings.shape[0] == len(input_data)
    return all_embeddings

def embed_align_on_the_fly(args, input_data, context_num):

    all_embeddings = []

    for ins_num, ins in tqdm(enumerate(input_data), total=len(input_data)):

        text = ins["text"].split("*****")[-1]
        text = text.lower()
        text = src.normalize_text.normalize(text)
        text_embedding = client.embeddings.create(input=[text], model="text-embedding-3-large").data[0].embedding
        all_embeddings.append(text_embedding)

    all_embeddings = torch.tensor(all_embeddings)
    torch.save(all_embeddings, args.output_file)
    return all_embeddings

def main(args, input_data, context_num):

    all_embeddings = embed_passages(args, input_data, context_num)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, default=None, help="Path to passages, eval_data path")
    parser.add_argument("--task", type=str, help="because it is data-specific")
    parser.add_argument("--output_file", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")

    args = parser.parse_args()

    input_data = []
    if args.input_file.endswith("jsonl"):
        with jsonlines.open(args.input_file) as f:
            for line in f:
                input_data.append(line)

    if args.task == "musique":
        context_num = 20
    else:
        context_num = 10

    main(args, input_data, context_num)
