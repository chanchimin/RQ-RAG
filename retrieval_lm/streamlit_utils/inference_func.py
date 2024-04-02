import copy
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

from data_curation.tools.duckduckgo_rapidapi import DDGSQueryRun

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def init_model_and_tokenizer_and_tool(model_name_or_path:str):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # this line is important for us to control when do we stop
    model.generation_config.eos_token_id = [tokenizer.convert_tokens_to_ids("</s>"),
                                            tokenizer.convert_tokens_to_ids("[EOS]")]

    model.generation_config.max_new_tokens = 512

    search_engine_api = DDGSQueryRun(3)

    return model, tokenizer, search_engine_api


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
        print("in async_wrapper, this query is '', might finish search, "
              "but for supporting batch decoding, still need to return [], do not have effect.")
        return []

    result = await loop.run_in_executor(None, search_engine_api, query)
    return result


# coroutine function
async def process_query_list(search_engine_api, query_list):
    tasks = [async_wrapper(search_engine_api, query) for query in query_list]
    results = await asyncio.gather(*tasks)
    return results


def generate_and_retrieve(examples, model, tokenizer, special_tokens_dict, search_engine_api,
                          search_limit, **kwargs):
    special_tokens_id = []
    for special in ["[S_Decomposed_Query]", "[S_Rewritten_Query]", "[S_Disambiguated_Query]", "[S_Response]",
                    "[A_Response]"]:
        special_tokens_id.append(special_tokens_dict[special])

    formatted_prediction = [""] * len(examples)
    raw_prediction = [""] * len(examples)

    # we are going to iterative generate until we reach the answer, but limit the search num
    search_nums = [0] * len(examples)
    is_done = [False] * len(examples)
    cur_examples = examples
    while True:
        # for llama tokenizer, it will always add <s> before the sentence,
        # if we do iterative generation we do not want this features, so add <s> in preprocess
        inputs = tokenizer(cur_examples, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        # intermediate_prediction = model.generate(**inputs, stopping_criteria=stopping_criteria)
        intermediate_prediction = model.generate(**inputs, logits_processor=kwargs["logits_processor"])

        # find what is before the stop word (batch).
        batch_special_index = []
        for in_batch_index, cur_ins in enumerate(intermediate_prediction):
            # that's the reason we do not change the generation_config.pad from 0 to 32...,
            # because the we want it to be zero in this case
            find_special_token = None
            special_index = len(cur_ins) - 1
            while special_index >= 0:
                # [... [EOS] ... ] we do not want to stop at EOS
                if cur_ins[special_index] in special_tokens_id and cur_ins[special_index] != tokenizer.convert_tokens_to_ids("[EOS]"):
                    find_special_token = cur_ins[special_index]
                    break
                special_index -= 1

            batch_special_index.append([find_special_token, special_index])

        # get_batch_query
        batch_queries = [""] * len(examples)
        batch_intermediate_string = tokenizer.batch_decode(intermediate_prediction)

        for in_batch_index, (find_special_token, special_index) in enumerate(batch_special_index):

            if find_special_token == tokenizer.convert_tokens_to_ids("[A_Response]"):
                # currently let model decide when to generate [A_response], add hard constraint afterwards
                # we generate the final answer
                if not batch_intermediate_string[in_batch_index].endswith("</s>"):
                    cur_example = batch_intermediate_string[in_batch_index] + "</s>"  # make sure to add this
                else:
                    cur_example = batch_intermediate_string[in_batch_index]
                cur_examples[in_batch_index] = cur_example.replace("<unk>", "").replace("<pad>", "")
                is_done[in_batch_index] = True
                continue
            elif find_special_token == tokenizer.convert_tokens_to_ids("[S_Response]"):
                # do not have effect, because this is the intermediate answer of the sub query, continue
                cur_example = tokenizer.decode(intermediate_prediction[in_batch_index])
                cur_examples[in_batch_index] = cur_example.replace("<unk>", "").replace("<pad>", "")
                continue

            elif find_special_token is None:
                # for mixed data training, sometimes do not trigger special token
                if not batch_intermediate_string[in_batch_index].endswith("</s>"):
                    cur_example = batch_intermediate_string[in_batch_index] + "</s>"  # make sure to add this
                else:
                    cur_example = batch_intermediate_string[in_batch_index]
                cur_examples[in_batch_index] = cur_example.replace("<unk>", "").replace("<pad>", "")
                is_done[in_batch_index] = True
                continue
            else:

                if is_done[in_batch_index]:
                    continue
                # now we assume that we meet the query token
                if search_nums[in_batch_index] >= search_limit:
                    # limit search time, do not use intermediate generated results
                    cur_examples[in_batch_index] += "[A_Response]"
                    continue

                search_nums[in_batch_index] += 1
                # we do not want the postfix <unk>, <EOS>
                query_for_search = tokenizer.decode(intermediate_prediction[in_batch_index][special_index + 1:-1],
                                                    skip_special_tokens=True)
                batch_queries[in_batch_index] = query_for_search

        if all(is_done):
            break

        evidences = asyncio.run(process_query_list(search_engine_api, batch_queries))
        evidences_list = format_evidences(evidences)

        for in_batch_index, in_batch_str in enumerate(batch_intermediate_string):
            if len(evidences_list[in_batch_index]) != 0:
                cur_examples[in_batch_index] = in_batch_str.replace("<unk>", "").replace("<pad>","")\
                                               + "[R_Evidences]" + evidences_list[in_batch_index] + "[/R_Evidences]"
            else:
                # actually we have jumped out from search_nums[in_batch_index] >= 1: , already add [A_Response]
                continue

        # yield every intermediate result here, because we are going to show it on the st.status
        yield {
                "search_query": batch_queries,
                "evidence_list": evidences_list,
                "cur_examples": cur_examples,
               }

    # extract the answer
    answers = []
    for cur in cur_examples:
        pattern = r"\[A_Response\](.*?)\[EOS\]"
        matches = re.findall(pattern, cur, re.DOTALL)
        if len(matches) > 0:
            result = matches[-1].strip()
            answers.append(result)
        else:

            # in this case, do not detect special token, it might direct generate the answer
            pattern2 = r"<\|assistant\|>(.*?)</s>"
            matches2 = re.findall(pattern2, cur, re.DOTALL)

            if len(matches2) > 0:
                result2 = matches2[-1].strip()
                answers.append(result2)
            else:
                answers.append("")

    return {"final_response": answers, "cur_examples": cur_examples}





