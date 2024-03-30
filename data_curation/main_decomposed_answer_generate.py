import datasets
import os
import json
import argparse
import random
from string import Template

import jsonlines

from llm_agent import LMAgent, \
    OpenAIRewriterLMAgent, \
    OpenAIJudgerLMAgent, \
    OpenAIGeneratorLMAgent, \
    OpenAIDecomposerLMAgent
from llm_agent.datatypes import State, Action
from llm_agent.prompt_template import DecomposeGeneratorTemplate

from tools.duckduckgo_rapidapi import DDGSQueryRun
from tools.bm25_candidates import BM25Run
from tqdm import tqdm

def call_DDGS(ins: dict, DDGS_agent: DDGSQueryRun, rewriter_agent=None, rewritten_queries=[], corpus =None, state:State = None):
    """
    directly return the top_k results

    :param ins:
    :return:
    retrieved_content
    query
    """

    original_query = ins["question"]

    if rewriter_agent is not None:
        response = rewriter_agent.act(original_query=original_query, rewritten_queries=rewritten_queries)
    else:
        response = original_query

    retrieved_evidences, retrieved_index = DDGS_agent(response, corpus)

    return retrieved_evidences, retrieved_index, response

def call_judge(ins: dict, original_query: str, retrieved_evidences: list, reference_output:str, judger_agent):
    """
    call the llm to give the reward on the query and retrieved results, (investigate the fine grained prompt later)

    :param ins:
    :return:
    """

    response = judger_agent.act(original_query=original_query, retrieved_evidences=retrieved_evidences, reference_output=reference_output)

    return response


def call_generator_decompose(ins: dict, prompt_template: str, decomposer_agent: OpenAIDecomposerLMAgent, **kwargs):

    """

    this function is used to generate the short answer for each unambiguous question;
    we take in the unambiguous question and the reference answer from dataset and ask llm to
    generate a reorganized answer if necessary.

    """

    kwargs["original_query"] = ins["question_text"]
    kwargs["type"] = ins["type"]
    if "evidences" in ins:
        kwargs["evidences"] = ins["evidences"]
    else:
        kwargs["evidences"] = None
    # kwargs["original_answers"] = ins["answer"]
    kwargs["original_answers"] = ins["answers_objects"][0]["spans"]

    contexts = ""
    for context in ins["contexts"]:
        contexts += f"Title: {context['title']}\nText: {context['paragraph_text']}"
        contexts += "\n"

    kwargs["contexts"] = contexts
    response = decomposer_agent.act(template=prompt_template,
                                   **kwargs)

    response = [r for r in response if len(r) != 0]

    return response


def call_generator_short(ins: dict, prompt_template: str, generator_agent, **kwargs):

    """

    this function is used to generate the short answer for each unambiguous question;
    we take in the unambiguous question and the reference answer from dataset and ask llm to
    generate a reorganized answer if necessary.

    """

    kwargs["original_query"] = ins["question"]
    kwargs["search_results"] = ins["search_results"]
    kwargs["original_answers"] = ins["short_answers"]

    response = generator_agent.act(template=prompt_template,
                                   **kwargs)

    return response

def call_generator_long(ins: dict, prompt_template: str, generator_agent, **kwargs):

    """

    similar to func: call_generator_short;
    this function is used to generate the long-form answer given all the unambiguous answer and its corresponding context
    and the llm are asked to generate a reorganized answer if necessary.

    the reason why we wrote 2 functions is because that not only the prompt template is different, but the logic and some information is also different

    """

    kwargs["ambiguous_question"] = ins["ambiguous_question"]

    unambiguous_answers = ""

    for index, cur_unambiguous in enumerate(ins["qa_pairs"]):
        unambiguous_answers += f"Unambiguous Question{index}: {cur_unambiguous['question']}\n"
        unambiguous_answers += f"Evidence{index}: {cur_unambiguous['search_results']} \n"
        unambiguous_answers += f"\n*** Answer{index}: {cur_unambiguous['gpt_responses']} ***\n\n"

    kwargs["unambiguous_questions_with_answers"] = unambiguous_answers

    original_long_form_answer = ""

    for index, long_form_answer in enumerate(ins["annotations"]):
        original_long_form_answer += f"{long_form_answer['long_answer']} \n"

    kwargs["original_answer"] = original_long_form_answer

    response = generator_agent.act(template=prompt_template, **kwargs)

    return response


def print_usage(agent_list:list):

    total_usage_profiles = {}

    for agent in agent_list:
        for usage_profile in agent.usage_profiles:
            for key, value in usage_profile.items():
                if key in total_usage_profiles:
                    total_usage_profiles[key] += value
                else:
                    total_usage_profiles[key] = value

    print("\n", total_usage_profiles)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path_decomposed', type=str, nargs="+", default=None, help="for decomposed dataset we assume the field is different from original ['instruction', 'input', 'output']")
    parser.add_argument('--ndocs', type=int, default=3, help="the number of retrieved evidences")
    parser.add_argument('--output_path', type=str, default="./generated_data/intermediate_data/2wikimultihopqa.json", help="the output path of our generated data")
    parser.add_argument('--n_samples', type=int, default=None,help="choose a fraction of number to create")
    parser.add_argument('--overwrite_output', default=False, action="store_true", help="decide whether to overwrite the outputdir")
    parser.add_argument('--api_type', type=str, default="openai", help="choose in openai, azure")
    parser.add_argument('--search_engine_type', type=str, default="duckduckgo")
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--openai_api_base', type=str, default=None)
    parser.add_argument('--model_version', type=str, default=None)
    args = parser.parse_args()

    openai_config = {
        "api_key": args.openai_api_key,
        "base_url": args.openai_api_base,
        "model_version": args.model_version
    }

    if args.search_engine_type == "duckduckgo":
        search_engine_api = DDGSQueryRun(max_results=args.ndocs)
    elif args.search_engine_type == "bm25_candidates":
        search_engine_api = BM25Run(args.ndocs)

    if args.api_type == "azure":

        # your config here
        azure_config = {
        }

        rewriter_agent = OpenAIRewriterLMAgent(api_type="azure", config=azure_config)
        judger_agent = OpenAIJudgerLMAgent(api_type="azure", config=azure_config)
        generator_agent = OpenAIGeneratorLMAgent(api_type="azure", config=azure_config)
        decomposer_agent = OpenAIDecomposerLMAgent(api_type="azure", config=azure_config)

    elif args.api_type == "openai":

        rewriter_agent = OpenAIRewriterLMAgent(api_type="openai", config=openai_config)
        judger_agent = OpenAIJudgerLMAgent(api_type="openai", config=openai_config)
        generator_agent = OpenAIGeneratorLMAgent(api_type="openai", config=openai_config)
        decomposer_agent = OpenAIDecomposerLMAgent(api_type="openai", config=openai_config)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    original_data = []
    for path in args.raw_data_path_decomposed:
        if path.endswith(".json"):
            cur_data = json.load(open(path))
            original_data.extend(cur_data)
        elif path.endswith(".jsonl"):
            cur_data = []
            with jsonlines.open(path) as f:
                for line in f:
                    cur_data.append(line)
            original_data.extend(cur_data)
        elif "hf_dataset" in path:
            cur_data = datasets.load_from_disk(path)["train"]
            cur_data_list = [item for item in cur_data]
            original_data.extend(cur_data_list)

    start_index = 0
    if os.path.exists(args.output_path) and not args.overwrite_output:

        with open(args.output_path) as f:
            generated_data = json.load(f)

        if not args.overwrite_output:
            print(f"Detect pre-existing data {len(generated_data)} samples, continue!")
            start_index = len(generated_data)

    else:
        generated_data = []

    if args.n_samples is not None:
        original_data = original_data[:args.n_samples]
    original_data = original_data[start_index:]

    for index, ins in tqdm(enumerate(original_data), total=len(original_data)):
        # continue to count index
        real_index = index + start_index

        # 1. for decomposed data like hotpotqa we will need to first ask the llm to decompose the query
        # 2. then used the decomposed query to query the search engine to get the evidences
        # 3. then use the llm to generate the answer conditioned on decomposed task

        # the returned value is only for reference, we will directly add results into 'ins' for further calling
        decomposed_queries: list = call_generator_decompose(ins, prompt_template=DecomposeGeneratorTemplate, decomposer_agent=decomposer_agent, state=None)

        ins["decomposed_queries"] = decomposed_queries
        ins["decomposed_queries_search_results"] = []
        ins["decomposed_queries_search_results_index"] = []

        # add search results to every unambiguous questions
        for cur_decomposed in decomposed_queries:
            cur_evidences, cur_retrieved_index, cur_query = call_DDGS({"question": cur_decomposed}, DDGS_agent=search_engine_api, corpus=ins["contexts"] , state=None)
            ins["decomposed_queries_search_results"].append(cur_evidences)
            ins["decomposed_queries_search_results_index"].append(cur_retrieved_index)

        # add llm_generated_answers conditioned on all the decomposed query

        print_usage([rewriter_agent, judger_agent, generator_agent, decomposer_agent])

        generated_data.append(ins)

        with open(args.output_path, "w") as w:
            json.dump(generated_data, w, indent=4)

    print("done")


if __name__ == '__main__':
    main()
