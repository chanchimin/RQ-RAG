import os
import json
import argparse
import random
import jsonlines

random.seed(42)

# A_Response is final answer, S_Response is intermediate answer to rewritten query
special_tokens_lists = [
    "[A_Response]",
    "[S_Rewritten_Query]",
    "[S_Decomposed_Query]",
    "[S_Disambiguated_Query]",
    "[R_Evidences]",
    "[/R_Evidences]",
    "[S_Response]",
    "[EOS]"
]

def add_space(ins:list) -> str:

    tmp = ""

    # ins contains list("title" "content")
    for example in ins:

        tmp += "Title: " + example["title"] + "\n"
        tmp += "Text: " + example["text"] + "\n"

    # clear the space at the end
    return tmp.strip()


nonsenses = ["The question you asked is not related to",
             "I cannot provide an answer",
             "that I can assist you with",
             "I am not capable of",
             "I cannot",
             ##
             "I'm sorry",
             "recommended answer",
             "is not related to",
             ##
             "InvalidRequestError",
             "As an AI",
             "original answer" 
             # "I cannot answer" # we do not filter this now, because some original data also contain this line
            ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge and add special token to the generated intermediate data")
    parser.add_argument("--raw_data_path", type=str, required=True, nargs="+", help="Input JSON datasets")
    parser.add_argument("--output_path", type=str, default="./generated_data/final_data/total.jsonl")
    parser.add_argument("--decomposed_ratio", type=float, default=1, help="2wiki have near 150k, do not need that much")
    parser.add_argument("--original_ratio", type=float, default=0.0, help="ablation that whether the original ratio will effect the final results")
    parser.add_argument("--no_search", default=False, action="store_true")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    original_data = []
    for path in args.raw_data_path:
        if path.endswith(".json"):
            cur_data = json.load(open(path))
            original_data.extend(cur_data)

    output_data = []
    for ins in original_data:
        # 1. deal with the data with the format of messages
        # because like arc, openbook, they all go into this entry, so there are a lot of unused key, like "answerKey"
        if "messages" in ins:
            formatted_ins = {}
            formatted_ins["id"] = ins["id"]
            formatted_ins["dataset"] = ins["dataset"]
            formatted_ins["messages"] = []
            for message_index, message in enumerate(ins["messages"]):

                # TODO deal with the role=="system" later
                if message["role"] == "assistant":
                    # because we store the query and evidence in role=="user", so we construct the
                    # response_with_special_token using the past message
                    response_with_special_token = ""

                    assert ins["messages"][message_index-1]["role"] == "user"
                    # need to search
                    if "all_queries" in ins["messages"][message_index-1]:

                        for index, (query, evidence) in enumerate(zip(ins["messages"][message_index-1]["all_queries"],
                                                                       ins["messages"][message_index-1]["all_evidences"])):


                            response_with_special_token += "[S_Rewritten_Query]"
                            response_with_special_token += "\n" + query + "\n"
                            response_with_special_token += "[EOS]"

                            response_with_special_token += "[R_Evidences]"
                            response_with_special_token += "\n" + add_space(evidence) + "\n"
                            response_with_special_token += "[/R_Evidences]"

                        response_with_special_token += "[A_Response]"

                        flag = False
                        for nonsense in nonsenses:
                            if "gpt_response" in message and nonsense in message["gpt_response"]:
                                flag = True
                                break
                        if flag or "answerKey" in ins:
                            response_with_special_token += "\n" + message["content"] + "\n"
                        else:
                            response_with_special_token += "\n" + message["gpt_response"] + "\n"

                        response_with_special_token += "[EOS]"

                    # do not need to search
                    else:
                        response_with_special_token += "[A_Response]"

                        flag = False
                        for nonsense in nonsenses:
                            if "gpt_response" in message and nonsense in message["gpt_response"]:
                                flag = True
                                break
                        if flag or "answerKey" in ins:
                            response_with_special_token += "\n" + message["content"] + "\n"
                        else:
                            if random.random() > args.original_ratio:
                                response_with_special_token += "\n" + message["gpt_response"] + "\n"
                            else:
                                response_with_special_token += "\n" + message["content"] + "\n"

                        response_with_special_token += "[EOS]"

                    if args.no_search:
                        formatted_ins["messages"].append({
                            "role": "assistant",
                            "content":  message["content"]
                        })
                    else:
                        formatted_ins["messages"].append({
                            "role": "assistant",
                            "content": response_with_special_token
                        })

                elif message["role"] == "user":
                    formatted_ins["messages"].append({
                        "role": "user",
                        "content": message["content"]
                    })

                elif message["role"] == "system":
                    formatted_ins["messages"].append({
                        "role": "system",
                        "content": message["content"]
                    })

            if "answerKey" in ins:
                assert formatted_ins["messages"][0]["role"] == "user"
                formatted_ins["messages"].insert(0, {
                    "role": "system",
                    "content": "Given four answer candidates, A, B, C and D, choose the best answer choice."
                })

            if formatted_ins["messages"][0]["role"] != "system":
                formatted_ins["messages"].insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant, please provide an answer to the questions, with the option to consider or ignore the provided context as needed."
                })
            output_data.append(formatted_ins)
        # 2. deal with the data without the format of messages, like decomposed and disambiguated
        else:

            formatted_ins = {}

            if "decomposed_queries" in ins:

                if args.decomposed_ratio is not None:
                    if random.random() >= args.decomposed_ratio:
                        continue

                messages = []

                formatted_ins["id"] = ins["question_id"]
                formatted_ins["dataset"] = ins["dataset"]
                response_with_special_token = ""

                for decomposed_index, (decomposed_query, decomposed_evidence) in enumerate(zip(ins["decomposed_queries"],
                                                                                         ins["decomposed_queries_search_results"])):
                    response_with_special_token += "[S_Decomposed_Query]"
                    response_with_special_token += "\n" + decomposed_query + "\n"
                    response_with_special_token += "[EOS]\n"

                    response_with_special_token += "[R_Evidences]"
                    response_with_special_token += "\n" + add_space(decomposed_evidence) + "\n"
                    response_with_special_token += "[/R_Evidences]\n"

                response_with_special_token += "[A_Response]"
                response_with_special_token += "\n" + "\n".join(ins["answers_objects"][0]["spans"]) + "\n"
                response_with_special_token += "[EOS]"

                messages.append({
                    "role": "system",
                    "content": "Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. Please provide a short and concise response."
                })

                messages.append({
                    "role": "user",
                    "content": f"{ins['question_text']}"
                })

                if args.no_search:
                    messages.append({
                        "role": "assistant",
                        "content": "\n".join(ins["answers_objects"][0]["spans"])
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": f"{response_with_special_token}"
                    })
                formatted_ins["messages"] = messages

            elif "ambiguous_question" in ins:

                messages = []

                formatted_ins["id"] = ins["sample_id"]
                formatted_ins["dataset"] = ins["dataset"]
                response_with_special_token = ""

                for disambiguated_index, disambiguated_qa_pairs in enumerate(ins["qa_pairs"]):
                    response_with_special_token += "[S_Disambiguated_Query]"
                    response_with_special_token += "\n" + disambiguated_qa_pairs["question"] + "\n"
                    response_with_special_token += "[EOS]\n"

                    response_with_special_token += "[R_Evidences]"
                    response_with_special_token += "\n" + add_space(disambiguated_qa_pairs["search_results"]) + "\n"
                    response_with_special_token += "[/R_Evidences]\n"

                    response_with_special_token += "[S_Response]"
                    # response_with_special_token += disambiguated_qa_pairs["gpt_responses"]
                    response_with_special_token += "\n" + "\n".join(disambiguated_qa_pairs["short_answers"]) + "\n"
                    response_with_special_token += "[EOS]\n"

                response_with_special_token += "[A_Response]"

                flag = False
                for nonsense in nonsenses:
                    if nonsense in ins["gpt_responses_long_form"]:
                        flag = True
                        break
                if flag:
                    response_with_special_token += "\n" + ins["annotations"][0]["long_answer"] + "\n"
                else:

                    if random.random() > args.original_ratio:
                        response_with_special_token += "\n" + ins["gpt_responses_long_form"] + "\n"
                    else:
                        response_with_special_token += "\n" + ins["annotations"][0]["long_answer"] + "\n"

                response_with_special_token += "[EOS]"

                messages.append({
                    "role": "system",
                    "content": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to disambiguate the question and provide a long-form answer including all correct answers."
                })

                messages.append({
                    "role": "user",
                    "content": f"{ins['ambiguous_question']}"
                })

                if args.no_search:
                    messages.append({
                        "role": "assistant",
                        "content": ins["annotations"][0]["long_answer"]
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": f"{response_with_special_token}"
                    })
                formatted_ins["messages"] = messages

            else:
                raise ValueError("not belong to multi_turn, decomposed, disambiguated")

            output_data.append(formatted_ins)

    print(f"total num: {len(output_data)}")
    # 3. save all the data with the messages format

    categories = {}
    for line in output_data:
        if line["dataset"] not in categories:
            categories[line["dataset"]] = 1
        else:
            categories[line["dataset"]] += 1

    print(categories)
    with jsonlines.open(args.output_path, "w") as writer:
        for line in output_data:
            assert len(line) == 3, "do not need other fields"
            writer.write(line)






