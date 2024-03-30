import datasets
import os

import json
import argparse
import random
from string import Template
from llm_agent import LMAgent, \
    OpenAIRewriterLMAgent, \
    OpenAIJudgerLMAgent, \
    OpenAIGeneratorLMAgent, \
    OpenAIMultiTurnLMAgent
from llm_agent.datatypes import State, Action
from llm_agent.prompt_template import MultiTurnGeneratorTemplate, MultiTurnGeneratorTemplateForAns, QAGeneratorTemplate
from tools.duckduckgo_rapidapi import DDGSQueryRun
from tqdm import tqdm

def call_DDGS(ins: dict, DDGS_agent: DDGSQueryRun, rewriter_agent=None, rewritten_queries=[], state:State = None):
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


    retrieved_evidences = DDGS_agent(response)

    return retrieved_evidences, response

def call_judge(ins: dict, original_query: str, retrieved_evidences: list, reference_output:str, judger_agent:ModelBestJudgerLMAgent):
    """
    call the llm to give the reward on the query and retrieved results, (investigate the fine grained prompt later)

    :param ins:
    :return:
    """

    response = judger_agent.act(original_query=original_query, retrieved_evidences=retrieved_evidences, reference_output=reference_output)

    return response


def call_generator_check_retrieval_rewritten(ins: dict, prompt_template: str, generator_agent, round_num: int, **kwargs):

    """

    this function is used to generate the short answer for each unambiguous question;
    we take in the unambiguous question and the reference answer from dataset and ask llm to
    generate a reorganized answer if necessary.

    """

    cur_history = ins["messages"][:round_num]
    history_str = ""

    for history in cur_history:
        history_str += str(history)
        history_str += "\n"

    query_str = ins["messages"][round_num]

    retrieval_necessity, queries = generator_agent.act(template=prompt_template,
                                   conversation_history=history_str,
                                   current_query=query_str)

    return retrieval_necessity, queries


def call_generator_for_ans(ins: dict, prompt_template: str, generator_agent, round_num: int, **kwargs):

    """

    similar to func: call_generator_short, long;
    this function is used to generate the long-form answer given all the rewritten query and its corresponding context
    and the llm are asked to generate a reorganized answer if necessary.

    the reason why we wrote 2 functions is because that not only the prompt template is different, but the logic and some information is also different

    """

    kwargs["cur_history_with_context"] = ins["messages"][:round_num-1]
    kwargs["current_query"] = ins["messages"][round_num-1]
    kwargs["recommended_answer"] = ins["messages"][round_num]

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
    parser.add_argument('--raw_data_path_multiturn', type=str, nargs="+", default=None, help="for ambiguous dataset we assume the field is different from original ['instruction', 'input', 'output']")
    parser.add_argument('--ndocs', type=int, default=3, help="the number of retrieved evidences")
    parser.add_argument('--output_path', type=str, default=None, help="the output path of our generated data")
    parser.add_argument('--n_samples', type=int, default=None,help="choose a fraction of number to create")
    parser.add_argument('--overwrite_output', default=False, action="store_true", help="decide whether to overwrite the outputdir")
    parser.add_argument('--api_type', type=str, default="openai", help="choose in openai, azure")
    parser.add_argument('--search_engine_type', type=str, default="duckduckgo")
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--openai_api_base', type=str, default=None)
    parser.add_argument('--is_qa', default=False, action="store_true")
    args = parser.parse_args()

    openai_config={
        "api_key": args.openai_api_key,
        "base_url": args.openai_api_base
    }

    if args.search_engine_type == "duckduckgo":
        search_engine_api = DDGSQueryRun(max_results=args.ndocs)

    if args.api_type == "azure":
        
        # your config
        azure_config = {
        }

        rewriter_agent = OpenAIRewriterLMAgent(api_type="azure", config=azure_config)
        judger_agent = OpenAIJudgerLMAgent(api_type="azure", config=azure_config)
        generator_agent = OpenAIGeneratorLMAgent(api_type="azure", config=azure_config)
        multiturn_agent = OpenAIMultiTurnLMAgent(api_type="azure", config=azure_config)

    elif args.api_type == "openai":

        rewriter_agent = OpenAIRewriterLMAgent(api_type="openai", config=openai_config)
        judger_agent = OpenAIJudgerLMAgent(api_type="openai", config=openai_config)
        generator_agent = OpenAIGeneratorLMAgent(api_type="openai", config=openai_config)
        multiturn_agent = OpenAIMultiTurnLMAgent(api_type="openai", config=openai_config)


    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    original_data = []
    for path in args.raw_data_path_multiturn:
        if path.endswith(".json"):
            with open(path) as f:
                cur_data = json.load(f)
            original_data.extend(cur_data)
        elif path.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    original_data.append(json.loads(line))

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

    is_qa = args.is_qa

    for index, ins in tqdm(enumerate(original_data), total=len(original_data)):
        # continue to count index
        real_index = index + start_index


        # 1. for multi-turn data like conversation we will need to iterate over the whole conversation and get the rewritten query and search results
        # 2. then generate a refined answer if necessary

        for round_num, round in enumerate(ins["messages"]):
            # do not need to do rewrite the query, but need to generate gpt answer
            if round["role"] == "assistant":

                if not is_qa:
                    cur_long_ans = call_generator_for_ans(ins,
                                                          prompt_template=MultiTurnGeneratorTemplateForAns,
                                                          generator_agent=generator_agent,
                                                          round_num=round_num)

                    # add llm_generated_long_form answer to ambiguous questions
                    round["gpt_response"] = cur_long_ans

                continue

            retrieval_necessity, queries = call_generator_check_retrieval_rewritten(ins,
                                                                                    prompt_template=MultiTurnGeneratorTemplate if not is_qa or "arc" in args.output_path else QAGeneratorTemplate,
                                                                                    generator_agent=multiturn_agent,
                                                                                    round_num=round_num)

            if retrieval_necessity:
                all_evidences = []
                all_queries = []


                for cur_query in queries:
                    cur_evidences, cur_query = call_DDGS({"question": cur_query.strip("-").strip(" ").strip("'")}, DDGS_agent=search_engine_api,
                                                         state=None)
                    all_evidences.append(cur_evidences)
                    all_queries.append(cur_query.strip("-").strip(" ").strip("'"))

                round["all_evidences"] = all_evidences
                round["all_queries"] = all_queries

        print_usage([rewriter_agent, judger_agent, multiturn_agent, generator_agent])

        generated_data.append(ins)

        with open(args.output_path, "w") as w:
            json.dump(generated_data, w, indent=4)



    print("done")


if __name__ == '__main__':
    main()
