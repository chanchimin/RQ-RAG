#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
import datasets
import logging
import os
import spacy
import jsonlines
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
# from vllm import LLM, SamplingParams
import random
import torch
import numpy as np
import openai
from tqdm import tqdm
import json
import argparse
import ast
import re
from tqdm import tqdm, trange
from collections import Counter
import string
import sys
import time
from utils import load_jsonlines, load_sag_special_tokens, preprocess_eval_data
from metrics import match, accuracy, match_batch, calculate_retrieval_em_f1
import datasets

# for adding the package data_creation_sag
sys.path.append("../")

# from data_creation_sag.tools.wikipedia_search import WikipediaQueryRun
from data_curation.tools.duckduckgo_rapidapi import DDGSQueryRun
from data_curation.tools.bingsearch_azure import BingSearchQueryRun
from data_curation.tools.bm25_candidates import BM25Run
from data_curation.tools.openai_embedding_search import OpenAIEmbedSearch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # print(input_ids)
        # [[1,2,3,0,0], [1,2,0,0,0]] ->
        # to find the first non-zero idx
        non_zero_inx = []
        for row in input_ids:
            index = len(row) - 1
            while index > 0:
                if row[index] != 0:
                    non_zero_inx.append(index)
                    break
                index -= 1
            non_zero_inx.append(index)

        # check if this position equal to stop ins
        for stop in self.stops:
            state = input_ids[:, non_zero_inx] == stop.repeat(input_ids.shape[0], 1)
            if torch.all(state).item():
                return True
        return False

def preprocess_input_data(dataset, task=None):
    new_data = []
    for item in dataset:
        # i do not use there line
        if "answers" not in item and "answerKey" in item:
            item["answers"] = [item["answerKey"]]

        if task in ["2wikimultihopqa", "hotpotqa", "musique"]:
            item["answers"] = []
            for obj in item["answers_objects"]:
                item["answers"].extend(obj["spans"])
        new_data.append(item)

    return new_data


def format_evidences(evidences: list):

    tmp = []

    for in_batch_evidences in evidences:
        ttmp = ""
        for cur_evidence in in_batch_evidences:
            ttmp += "Title: " + cur_evidence["title"] + "\n"
            ttmp += "Text: " + cur_evidence["text"] + "\n"

        tmp.append(ttmp)

    return tmp


async def async_wrapper(search_engine_api, query):
    loop = asyncio.get_event_loop()

    if query == "":
        print("in async_wrapper, this query is '', might finish search, return []")
        return []

    result = await loop.run_in_executor(None, search_engine_api, query)
    return result


# coroutine function
async def process_query_list(search_engine_api, query_list):
    tasks = [async_wrapper(search_engine_api, query) for query in query_list]
    results = await asyncio.gather(*tasks)
    return results

def generate_tree_of_thoughts(model, tokenizer, initial_prompts, raw_datas, special_tokens_dict, max_depth, max_width,
                              search_engine_api, search_limit, args, index, total_corpus):

    expand_on_tokens = args.expand_on_tokens

    paths = [{'prompt': prompt, 'depth': 0, 'text': '', 'done': False, "retrieved_index": []} for prompt in initial_prompts]
    final_outputs = []

    while paths:
        current_path = paths.pop(0)

        if current_path['done']:
            final_outputs.append(current_path)
            continue

        for special_token in expand_on_tokens:

            if current_path['depth'] >= max_depth and special_token != "[A_Response]":
                continue

            input_text = current_path['prompt'] + special_token

            inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)

            if special_token != "[A_Response]":
                # when not generating the final answer, we adjust the temp to increase diversity
                outputs = model.generate(**inputs, return_dict_in_generate=True, temperature=1.0)
            else:
                outputs = model.generate(**inputs, return_dict_in_generate=True, do_sample=False)

            decoded_output = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0].replace("<s> ", "<s>")

            if special_token == "[A_Response]":
                # done
                pattern = r"\[A_Response\](.*?)\[EOS\]"
                matches = re.findall(pattern, decoded_output, re.DOTALL)
                if len(matches) > 0:
                    result = matches[-1].strip()
                else:
                    result = "dummy results, unable to detect valid answer"

                new_path = {
                    'prompt': decoded_output,
                    'depth': current_path['depth'] + 1,
                    'done': True,
                    'final_answer': result,
                    'retrieved_index': current_path['retrieved_index'],
                }
            else:
                # get the query and ask search_engine
                pattern = r"\[(S_Rewritten_Query|S_Decomposed_Query|S_Disambiguated_Query)\](.*?)\[EOS\]"
                matches = re.findall(pattern, decoded_output, re.DOTALL)
                if len(matches) > 0:
                    query_for_search = matches[-1][1].strip()
                else:
                    query_for_search = "dummy"

                if args.task in ["2wikimultihopqa", "hotpotqa", "musique"]:
                    if args.search_engine_type == "openai_embed":
                        evidences, top_indices = search_engine_api(query_for_search, corpus=raw_datas[0]["contexts"], index=index)
                    else:
                        evidences, top_indices = search_engine_api(query_for_search, corpus=raw_datas[0]["contexts"])
                    evidences_list = format_evidences([evidences])

                else:
                    top_indices = [None] # use search engine, do not have this
                    evidences = search_engine_api(query_for_search)
                    evidences_list = format_evidences([evidences])

                new_path = {
                    'prompt': decoded_output + "[R_Evidences]" + evidences_list[0] + "[/R_Evidences]",
                    'depth': current_path['depth'] + 1,
                    'done': False,
                    'cur_special_token': special_token,
                    'cur_query': query_for_search,
                    'cur_evidence': evidences_list[0],
                    'retrieved_index': current_path['retrieved_index'] + top_indices,
                }

            paths.append(new_path)

    if args.oracle:
        # it means that we concat all the anwers as the final answer
        # our logic here is that , among all the candidates, if there exist the answer, then it is correct
        # then afterwards, we have to find how to aggregate the answer
        pred = ""
        for current_path in final_outputs:
            pred += current_path["final_answer"] + "\n"

    # support the same api, view it as batch encoding
    return [pred], [final_outputs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="bfloat16",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `float16`.")
    # Decoding hyperparams
    parser.add_argument('--metric', default="match", type=str, help="metric to be used during evaluation")
    parser.add_argument('--use_search_engine', default=False, action="store_true")
    parser.add_argument('--overwrite_output_dir', default=False, action="store_true")
    parser.add_argument('--use_hf', default=False, action="store_true", help="choose whether to use huggingface for inference or use vllm")
    parser.add_argument('--task', type=str, help="which benchmark are we now testing")
    parser.add_argument('--use_flash_attn', default=False, action="store_true", help="whether to use flash_attn, but we have to check whether it will cause much difference")
    parser.add_argument('--search_limit', type=int, default=1,
                        help="whether to limit search")
    parser.add_argument('--inference_batch_size', type=int, default=1,
                        help="actually for the agent style inference, it typically support batch size == 1")
    parser.add_argument('--tree_decode', default=False, action="store_true")
    parser.add_argument('--max_depth', type=int, default=2, help="pratically usefull when inference 2wikihop, may retrieve up to 4 times")
    parser.add_argument('--oracle', default=False, action="store_true", help="it means that we are testing the upperbound of the system, where we assume all candidates are valid.")
    parser.add_argument('--search_engine_type', type=str, required=True, default=None)
    parser.add_argument('--rapidapi_name', type=str, default="one", help="custom rapid api ")
    parser.add_argument('--expand_on_tokens', nargs="+", help="do not want to expand on all, we have three different special tokens")

    args = parser.parse_args()

    if args.use_flash_attn:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    input_path = args.input_file
    if input_path.endswith(".json"):
        with open(input_path) as f:
            input_data = json.load(f)
    elif input_path.endswith(".jsonl"):
        input_data = load_jsonlines(input_path)
    else:
        input_data = datasets.load_dataset(input_path)["validation"]

    total_corpus = None
    # TODO to make it more easy to use using argument.
    if args.search_engine_type == "duckduckgo":
        search_engine_api = DDGSQueryRun(args.ndocs, args.rapidapi_name)
    elif args.search_engine_type == "bingsearch":
        search_engine_api = BingSearchQueryRun(args.ndocs)
    elif args.search_engine_type == "bm25_candidates":
        search_engine_api = BM25Run(args.ndocs)
    elif args.search_engine_type == "openai_embed":
        search_engine_api = OpenAIEmbedSearch(args.ndocs, task=args.task, args=args)

    os.makedirs(args.output_path, exist_ok=True)

    all_preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    start_index = 0

    if not args.overwrite_output_dir:
        if os.path.exists(os.path.join(args.output_path, "final_results.json")):
            raise ValueError(
                "final_results.json exists, if you sure you want to rerun, please specify --overwrite_output_dir")
        elif os.path.exists(os.path.join(args.output_path, "tmp_results.json")):
            with open(os.path.join(args.output_path, "tmp_results.json"), "r") as f:
                tmp_results = json.load(f)
            all_preds = tmp_results["all_preds"]
            prompts = tmp_results["prompts"]
            metric_results = tmp_results["metric_results"]
            golds = tmp_results["golds"]
            # scores = tmp_results["scores"]
            all_results = tmp_results["all_results"]
            metric_mean = tmp_results["metric_mean"]
            metric = tmp_results["metric"]

            logging.warning(f"Already done {len(tmp_results['all_preds'])} instances, continue!")
            start_index = len(tmp_results['all_preds'])


    input_data = preprocess_input_data(
        input_data, task=args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    # Remember to load from .pt, or it will automatically load from safe_tensor which cause error

    special_tokens_dict = load_sag_special_tokens(tokenizer)

    if args.use_hf:

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

        model.generation_config.eos_token_id = [tokenizer.convert_tokens_to_ids("</s>"),
                                                tokenizer.convert_tokens_to_ids("[EOS]")]
        model.generation_config.max_new_tokens = args.max_new_tokens

    stop_words = ["[EOS]"]
    stop_words_ids = [
        tokenizer.encode(stop_word, add_special_tokens=False, return_tensors='pt').squeeze(0).to(model.device) for
        stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    inference_batch_size = args.inference_batch_size

    for i in trange(0, len(input_data[start_index:]), inference_batch_size):

        rows = input_data[start_index:][i: i+inference_batch_size]

        eval_datas = preprocess_eval_data(rows, tokenizer=tokenizer, task=args.task)
        if args.tree_decode:

            assert len(eval_datas) == 1, "tree decoding only support batch size == 1"

            preds, meta_results = generate_tree_of_thoughts(
                model=model,
                tokenizer=tokenizer,
                initial_prompts=eval_datas,
                special_tokens_dict=special_tokens_dict,
                max_depth=args.max_depth,
                max_width=3,
                search_engine_api=search_engine_api,
                search_limit=args.search_limit,
                args=args,
                index=start_index+i,
                total_corpus=total_corpus,
                raw_datas=rows,)

        for row in rows:
            if "answers" not in row and "answer" in row:
                row["answers"] = [row["answer"]] if type(
                    row["answer"]) is str else row["answer"]

        if args.task in ["2wikimultihopqa", "hotpotqa", "musique"]:
            # get the overall retrieval performance EM and f1 of differenct path
            gold_support_idxs = []
            for index, context in enumerate(rows[0]["contexts"]):
                if context["is_supporting"]:
                    gold_support_idxs.append(index)
            if args.search_engine_type != "elastic_search":
                # for retrieve from candidates setting, we can calc retrieval performance
                for path in meta_results[0]:
                    predicted_support_idxs = path['retrieved_index']
                    cur_f1, cur_em  = calculate_retrieval_em_f1(predicted_support_idxs, gold_support_idxs)
                    path["retrieval_performance"] = [cur_f1, cur_em]

        if "arc" in args.task or "openbookqa" in args.task:
            label_dict = {
                "A": "A",
                "B": "B",
                "C": "C",
                "D": "D",
                "E": "E",
                "1": "A",
                "2": "B",
                "3": "C",
                "4": "D",
                "5": "E"
            }
            golds.extend([row["answerKey"] for row in rows])
            for row in rows:
                row["answers"] = [label_dict[row["answerKey"]],
                                  row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])]]

        else:
            golds.extend([row["answers"] if "output" not in row else row["output"] for row in rows])

        prompts.extend(eval_datas)
        all_preds.extend(preds)
        all_results.extend(meta_results)

        if args.metric == "accuracy":
            metric_result = accuracy(preds, rows)

        elif args.metric == "match":
            metric_result = match_batch(preds, rows)
        else:
            raise NotImplementedError

        metric_results.extend(metric_result)
        if i % (2 * inference_batch_size) == 0:
            print("average: {}".format(np.mean(metric_results)))
            final_results = {"all_preds": all_preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                             "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results),
                             }
            with open(os.path.join(args.output_path, "tmp_results.json"), "w") as outfile:
                json.dump(final_results, outfile)

    final_results = {"all_preds": all_preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                     "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results),
                     }
    with open(os.path.join(args.output_path, "final_results.json"), "w") as outfile:
        json.dump(final_results, outfile)

    print("Final result: {0}".format(np.mean(metric_results)))
    print("Retrieval Frequencies: {0}".format(count / len(final_results["all_preds"])))


if __name__ == "__main__":
    main()
