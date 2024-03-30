import jsonlines
import json
import copy
import re

system_prompt = {
    "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "popqa_longtail_w_gs": "Please answer the question.",
    "openbookqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "hotpotqa": "Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. Please provide a short and concise response.",
    "2wikimultihopqa":"Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. Please provide a short and concise response.",
    "musique": "Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. Please provide a short and concise response.",
}


def preprocess_single_data(user_query, tokenizer, previous_history=""):

    user_query = "<s><|user|>\n" + user_query + tokenizer.eos_token + "\n"
    user_query += "<|assistant|>\n"

    return [user_query]

def preprocess_multi_turn_inference_data(user_query, previous_history, tokenizer, task, turn, retrieve_type):

    # apparently multi_turn inference is hard to support batch decoding.

    if task in ["mt_bench", "alpaca_eval"]:

        if previous_history == "":
            eval_example = "<s>"
        else:
            eval_example = ""
            previous_history += "\n"
        eval_example += previous_history + "<|user|>\n" + user_query[turn] + tokenizer.eos_token + "\n"

        if retrieve_type == "always":
            eval_example += "<|assistant|>\n[S_Rewritten_Query]"
        elif retrieve_type == "never":
            eval_example += "<|assistant|>\n[A_Response]"
        elif retrieve_type == "pure":
            eval_example += "<|assistant|>\n"

    else:
        raise ValueError(f"task {task} do not in the lists")

    return [eval_example]




def preprocess_eval_data(row, tokenizer, task):

    eval_examples = []

    for cur in row:
        if task in ["popqa_longtail_w_gs", "popqa_longtail_w_gs_may_refers_to"]:
            eval_example = f"<s><|system|>\n{system_prompt[task]}" + tokenizer.eos_token + "\n<|user|>\n" + cur["question"] + tokenizer.eos_token + "\n"
            eval_example += "<|assistant|>\n"

        elif task in ["hotpotqa", "2wikimultihopqa", "musique"]:
            eval_example = f"<s><|system|>\n{system_prompt[task]}" + tokenizer.eos_token + "\n<|user|>\n" + cur["question_text"] + tokenizer.eos_token + "\n"
            eval_example += "<|assistant|>\n"

        elif task == "arc_challenge":
            label_dict = {
                "A": "A",
                "B": "B",
                "C": "C",
                "D": "D",
                "1": "A",
                "2": "B",
                "3": "C",
                "4": "D",
            }

            if cur["question"].endswith("?"):
                user_query = f"{cur['question']}"
            else:
                user_query = f"{cur['question']}?"

            user_query += "\n"
            user_query += "Please choose from following options:\n"

            for option_index, option_text in enumerate(cur["choices"]["text"]):
                if cur["choices"]["label"][option_index] not in label_dict:
                    print(cur["choices"]["label"][option_index])
                    continue

                user_query += "{0}: {1}\n".format(label_dict[cur["choices"]["label"][option_index]], option_text)

            eval_example = f"<s><|system|>\n{system_prompt[task]}" + tokenizer.eos_token + "\n<|user|>\n" + user_query + tokenizer.eos_token + "\n"
            eval_example += "<|assistant|>\n"

        elif task == "openbookqa":

            label_dict = {
                "A": "A",
                "B": "B",
                "C": "C",
                "D": "D",
                "1": "A",
                "2": "B",
                "3": "C",
                "4": "D",
            }

            if cur["question_stem"].endswith("?"):
                user_query = f"{cur['question_stem']}"
            else:
                user_query = f"{cur['question_stem']}?"

            user_query += "\n"
            user_query += "Please choose from following options:\n"

            for option_index, option_text in enumerate(cur["choices"]["text"]):
                if cur["choices"]["label"][option_index] not in label_dict:
                    print(cur["choices"]["label"][option_index])
                    continue

                user_query += "{0}: {1}\n".format(label_dict[cur["choices"]["label"][option_index]], option_text)

            eval_example = f"<s><|system|>\n{system_prompt[task]}" + tokenizer.eos_token + "\n<|user|>\n" + user_query + tokenizer.eos_token + "\n"
            eval_example += "<|assistant|>\n"

        else:
            raise ValueError(f"task {task} do not in the lists")

        eval_examples.append(eval_example)

    return eval_examples


def load_sag_special_tokens(tokenizer):

    special_tokens_dict = {}

    for token in tokenizer.additional_special_tokens:
        special_tokens_dict[token] = tokenizer.convert_tokens_to_ids(token)

    return special_tokens_dict


def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)
