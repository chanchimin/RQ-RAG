import json
import jsonlines
import os
import torch
import argparse
import transformers
from tqdm import trange
import logging
import re
import datasets
import string
import collections
import ftfy
import wandb
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

label_dict = {
        "A": ["A", "▁A"],
        "B": ["B", "▁B"],
        "C": ["C", "▁C"],
        "D": ["D", "▁D"],
        "E": ["E", "▁E"],
        "1": ["A", "▁A"],
        "2": ["B", "▁B"],
        "3": ["C", "▁C"],
        "4": ["D", "▁D"],
        "5": ["E", "▁E"]
    }

def calculate_retrieval_recall(predicted_support_idxs, gold_support_idxs, is_title=False):
    # Taken from hotpot_eval

    if is_title:
        cur_sp_pred = set(predicted_support_idxs)
        gold_sp_pred = set(gold_support_idxs)
    else:
        cur_sp_pred = set(map(int, predicted_support_idxs))
        gold_sp_pred = set(map(int, gold_support_idxs))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0

    return recall


def get_prompt_ppl_and_retrieve_confidence(model, tokenizer, ins, options, is_qa):

    prompt = ins.split("<|assistant|>")[0] + "<|assistant|>"

    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False, padding=False).to(model.device)
    with torch.no_grad():
        output = model(**inputs, labels=inputs["input_ids"])

    prompt_ppl = torch.exp(output.loss).item()

    logits = output.logits

    special_tokens_ids = []
    for special_tokens in ["[A_Response]", "[S_Rewritten_Query]", "[S_Decomposed_Query]", "[S_Disambiguated_Query]"]:
        special_tokens_id = tokenizer.convert_tokens_to_ids(special_tokens)
        special_tokens_ids.append(special_tokens_id)

    special_tokens_logits = logits[0, -1, special_tokens_ids]
    probabilities = torch.softmax(special_tokens_logits, dim=-1)

    retrieve_confidence = 1 - probabilities[0].item()

    return prompt_ppl, retrieve_confidence

def get_ppl_and_answer_confidence_and_option(model, tokenizer, ins, options, is_qa):

    inputs = tokenizer([ins], return_tensors="pt", add_special_tokens=False, padding=False).to(model.device)

    if is_qa:
        answer_pattern = r"(?:is )([A-E])(?:\:|\.|\s+|\[EOS\])"
    else:
        answer_pattern = r"(?:is )(.*?)(?:\[EOS\])"
    matches = re.findall(answer_pattern, ins.split("[A_Response]")[-1], re.DOTALL)
    if matches:
        extracted_result = matches[-1]
    else:
        extracted_result = None

    with torch.no_grad():
        output = model(**inputs, labels=inputs["input_ids"])

    # get confidence
    start_token = "[A_Response]"
    end_token = "[EOS]"
    inputs = tokenizer.encode(ins, return_tensors="pt", add_special_tokens=False)

    start_pos = (inputs == tokenizer.encode(start_token, add_special_tokens=False)[0]).nonzero(as_tuple=True)[1][-1:] # although not preferable, sometimes it does emerge multiple time
    end_pos = (inputs == tokenizer.encode(end_token, add_special_tokens=False)[0]).nonzero(as_tuple=True)[1][-1:] # there might be multiple [EOS]
    logits = output.logits
    # the next token id is 13 which is "\n", and do not know why sometimes have two \n, and it's different from final answer
    while start_pos < len(inputs[0]) - 1  and inputs[0][start_pos].item() in [32000, 29871,13]:
        start_pos += 1

    if len(start_pos) == 0:
        start_pos = 0
    if len(end_pos) == 0:
        end_pos = 0

    if is_qa:
        selected_logits = logits[0, start_pos-1:start_pos, :] # [A_Response] ** only this one ** [EOS], should be aware, the start_pos in logits means the first output
    else:
        selected_logits = logits[0, start_pos - 1:end_pos - 1, :]
    # get answers after [A_Response] configence
    probabilities = torch.softmax(selected_logits, dim=-1)
    max_confidence, _ = torch.max(probabilities, dim=-1)
    average_confidence = torch.mean(max_confidence).item()

    # get ppl
    ppl = torch.exp(output.loss).item()

    if options is not None:
        # get options logits
        option_logits_ids = []
        for option in options:
            option_id = tokenizer.convert_tokens_to_ids(option)
            option_logits_ids.append(option_id)

        option_logits = torch.softmax(logits[0, start_pos-1, option_logits_ids],dim=-1)
        max_index = torch.argmax(option_logits).item()
        selected_option = options[max_index]
        selected_option_score = option_logits[max_index].item()
    else:
        selected_option = extracted_result # might be none
        selected_option_score = average_confidence

    return ppl, average_confidence, selected_option, selected_option_score, extracted_result

def match(prediction, ground_truth):
    for gt in ground_truth:

        if not isinstance(gt, str) or not isinstance(prediction, str):
            print(f"prediction : {prediction}")
            print(f"gt : {gt}")
            return 0

        if gt.lower() in prediction.lower():
            return 1
    return 0

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_pred, a_gold):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):

    if prediction is None or len(prediction) == 0:
        return 0

    prediction = ftfy.fix_text(prediction)
    ground_truths = [ftfy.fix_text(e) for e in ground_truths]

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help="path where we store the whole hidden state")
    parser.add_argument('--task', type=str, help="what task you are eval now?")
    parser.add_argument('--original_data', type=str, help="in order to get choice", default=None)
    parser.add_argument('--model_name_or_path', type=str, help="because we need to get ppl, so we need the score")
    parser.add_argument('--calc_retrieval_performance', default=False, action="store_true")
    parser.add_argument('--from_elastic_search', default=False, action="store_true")
    parser.add_argument('--calc_depth', type=int, nargs="+", default=None, help="If none, calc all")
    parser.add_argument('--calc_width', type=str, nargs="+", default=None, help="If none, calc all")
    parser.add_argument('--retrieve_threshold', default=0.0, type=float,
                        help="if retrieve confidence lower than this threshold, use no_retrieve results only")
    parser.add_argument('--prompt_ppl_threshold', default=100, type=float,
                        help="not used currently")
    parser.add_argument('--use_selected_token', default=False, action="store_true")

    args = parser.parse_args()

    if args.original_data is not None:
        gold_support_idxs = []
        if args.original_data.endswith("json"):
            with open(args.original_data) as f:
                original_data = json.load(f)
        elif args.original_data.endswith(".jsonl"):
            original_data = []
            with jsonlines.open(args.original_data) as f:
                for line in f:
                    original_data.append(line)
        else:
            original_data = []

            # load from hf dataset
            raw_datasets = datasets.load_dataset(
                args.original_data,
            )
            for line in raw_datasets["validation"]:
                original_data.append(line)

        if args.calc_retrieval_performance:
            for line in original_data:
                gold_support_idx = []
                for index, context in enumerate(line["contexts"]):
                    if context["is_supporting"]:
                        if args.from_elastic_search:
                            gold_support_idx.append(context["title"])
                        else:
                            gold_support_idx.append(index)
                gold_support_idxs.append(gold_support_idx)

    else:
        original_data = None

    # read results
    with open(os.path.join(args.run_name, "final_results.json")) as f:
        results = json.load(f)

    total_num = len(results["prompts"])

    assert len(original_data) == total_num

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                   device_map="auto",
                                                   torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                                   )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    results["score_lists"] = []

    oracle_accurate_num = 0
    ppl_accurate_num = 0
    confidence_accurate_num = 0
    ensemble_accurate_num = 0

    total_max_f1 = 0
    total_ppl_f1 = 0
    total_confidence_f1 = 0
    total_ensemble_f1 = 0

    total_max_retrieval_performance = [0, 0]
    total_ppl_retrieval_performance = [0, 0]
    total_confidence_retrieval_performance = [0, 0]
    total_ensemble_retrieval_performance = [0, 0]

    total_max_retrieval_recall = 0
    total_ppl_retrieval_recall = 0
    total_confidence_retrieval_recall = 0
    total_ensemble_retrieval_recall = 0

    manual_check = defaultdict(list)

    if "arc" in args.task or "openbookqa" in args.task:
        options = ["A", "B", "C", "D"]
        is_qa = True
    elif "2wikimultihopqa" in args.task or "hotpotqa" in args.task or "musique" in args.task or "popqa" in args.task:
        options = None
        is_qa = False

    for index in trange(total_num):
        cur_scores_list = []
        oracle_success = False
        max_f1 = -1
        max_f1_ins = results["all_results"][index][0] # the first non retrieve path
        row = original_data[index]

        if "choices" in row:
            # for the multichoice qa dataset
            literal_answer = row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])]
        elif "answers_objects" in row:
            # multihop
            literal_answer = row["answers_objects"][0]["spans"]
        elif "answers" in row:
            # for popqa
            literal_answer = row["answers"]
        elif "answer" in row:
            # for popqa
            literal_answer = row["answer"]

        if isinstance(literal_answer, str) and literal_answer.endswith("."):
            literal_answer = literal_answer[:-1]

        answer2score = {}
        fever_answer2score = {}
        used_path = []
        for path_index in range(len(results["all_results"][index])):

            if args.calc_depth is not None and results["all_results"][index][path_index]["depth"] not in args.calc_depth:
                continue

            matches = re.findall(r'S_Rewritten_Query|S_Decomposed_Query|S_Disambiguated_Query|A_Response', results["all_results"][index][path_index]["prompt"])

            if args.calc_width is not None:
                if len(matches) > 1:
                    # for the case the output is not what we like might [s1,s2,a,a]
                    def remove_specific_duplicates(lst, target):
                        seen = False
                        result = []
                        for item in lst:
                            if item == target:
                                if not seen:
                                    result.append(item)
                                    seen = True
                            else:
                                result.append(item)
                        return result

                    skip = False
                    detected_special = remove_specific_duplicates(matches, "A_Response")
                    for item in detected_special:
                        if item not in args.calc_width:
                            skip = True

                    if skip:
                        continue
                else:
                    if matches[-1] not in args.calc_width:
                        continue

            cur_scores = {}
            ppl, confidence, selected_option, selected_option_score, extracted_result = get_ppl_and_answer_confidence_and_option(model, tokenizer, results["all_results"][index][path_index]["prompt"], options, is_qa)
            if args.use_selected_token:
                results["all_results"][index][path_index]["final_answer"] = selected_option
            final_answer = results["all_results"][index][path_index]["final_answer"]
            results["all_results"][index][path_index]["ppl"] = ppl
            results["all_results"][index][path_index]["confidence"] = confidence
            results["all_results"][index][path_index]["extracted_result"] = extracted_result

            if extracted_result is not None:
                selected_option = extracted_result

            if selected_option is None:
                selected_option = final_answer

            if selected_option != final_answer:
                print(f"selected_option {selected_option} != final_answer {final_answer}")

            results["all_results"][index][path_index]["selected_option"] = selected_option
            results["all_results"][index][path_index]["selected_option_score"] = selected_option_score

            cur_scores["ppl"] = ppl
            cur_scores["confidence"] = confidence
            cur_scores_list.append(cur_scores)

            if is_qa and selected_option not in options:
                raise ValueError("check selected_option")

            if is_qa:
                if final_answer in label_dict[results["golds"][index]]:
                    oracle_success = True

            elif "popqa" in args.task:
                # specially handle this task, not qa, no retrieval performance
                if match(final_answer, results["golds"][index]):
                    oracle_success = True

            else:
                cur_f1 = metric_max_over_ground_truths(compute_f1, results["all_results"][index][path_index]["final_answer"], literal_answer)

                if cur_f1 > max_f1:
                    # update f1 and f1_ins
                    max_f1 = cur_f1
                    max_f1_ins = results["all_results"][index][path_index]
                elif cur_f1 == max_f1:
                    # only update max_f1_ins when retrieval_performance is also higher

                    if "retrieval_performance" in results["all_results"][index][path_index]:

                        if len(results["all_results"][index][path_index]["retrieval_performance"]) > 0 and \
                                results["all_results"][index][path_index]["retrieval_performance"][0] > \
                                max_f1_ins["retrieval_performance"][0]:
                            max_f1_ins = results["all_results"][index][path_index]

                if match(results["all_results"][index][path_index]["final_answer"], literal_answer):
                    oracle_success = True

            if final_answer not in answer2score:
                answer2score[final_answer] = 0

            answer2score[final_answer] += confidence

            # first get the retrieve confidence to decide whether to calc the retrieve results
            if path_index == 0:
                prompt_ppl, retrieve_confidence = get_prompt_ppl_and_retrieve_confidence(model, tokenizer,
                                                                                         results["all_results"][index][
                                                                                             path_index]["prompt"],
                                                                                         options, is_qa)

                # only used the non retrieve results
                if retrieve_confidence < args.retrieve_threshold:
                    used_path = [0]
                    break

            used_path.append(path_index)

        used_ins = [results["all_results"][index][i] for i in used_path]

        if "2wikimultihopqa" in args.task or "hotpotqa" in args.task or "musique" in args.task:
            total_max_f1 += max_f1
            # get the max ins' retrieval performance, [f1, em]

            if not args.from_elastic_search:

                cur_max_retrieval_performance = max_f1_ins["retrieval_performance"]
                total_max_retrieval_performance = [cur + pre for cur, pre in zip(cur_max_retrieval_performance,
                                                                                 total_max_retrieval_performance)]

                cur_max_retrieval_recall = calculate_retrieval_recall(max_f1_ins["retrieved_index"], gold_support_idxs[index])
                total_max_retrieval_recall += cur_max_retrieval_recall

            else:
                cur_max_retrieval_performance = [0, 0]
                total_max_retrieval_performance = [cur + pre for cur, pre in zip(cur_max_retrieval_performance,
                                                                                 total_max_retrieval_performance)]

                cur_max_retrieval_recall = calculate_retrieval_recall(max_f1_ins["retrieved_index"],
                                                                      gold_support_idxs[index],
                                                                      is_title=True)
                total_max_retrieval_recall += cur_max_retrieval_recall

        if oracle_success:
            oracle_accurate_num += 1
        else:
            print(index)

        results["score_lists"].append(cur_scores_list)

        # pick the option with the maximum score
        sorted_answers = sorted(
            answer2score.items(), key=lambda x: x[1], reverse=True)
        ensemble_ans = sorted_answers[0][0]
        if is_qa:
            if ensemble_ans in label_dict[results["golds"][index]]:
                ensemble_accurate_num += 1

        elif "popqa" in args.task:
            # specially handle this task, not qa, no retrieval performance
            if match(ensemble_ans, results["golds"][index]):
                ensemble_accurate_num += 1
        else:
            # traverse the path to find the ans that equal to the ensemble_ans and get the retrieval performance
            cur_avg_ensemble_retrieval_performance = [0, 0]
            cur_avg_ensemble_retrieval_recall = 0
            ensemble_num = 0
            for path_index in range(len(used_ins)):
                if used_ins[path_index]["final_answer"] == ensemble_ans:
                    ensemble_num += 1

                    if not args.from_elastic_search:
                        cur_ensemble_retrieval_performance = used_ins[path_index]["retrieval_performance"]

                        cur_avg_ensemble_retrieval_performance = [cur + pre for cur, pre in zip(cur_ensemble_retrieval_performance,
                                                                                                cur_avg_ensemble_retrieval_performance)]
                        cur_avg_ensemble_retrieval_recall += calculate_retrieval_recall(used_ins[path_index]["retrieved_index"],
                                                                      gold_support_idxs[index])
                    else:

                        cur_avg_ensemble_retrieval_recall += calculate_retrieval_recall(
                            used_ins[path_index]["retrieved_index"],
                            gold_support_idxs[index],
                            is_title=True)

            cur_avg_ensemble_retrieval_performance = [cur/ensemble_num for cur in cur_avg_ensemble_retrieval_performance]
            cur_avg_ensemble_retrieval_recall = cur_avg_ensemble_retrieval_recall/ensemble_num

            total_ensemble_retrieval_performance = [cur + pre for cur, pre in zip(cur_avg_ensemble_retrieval_performance,
                                                                             total_ensemble_retrieval_performance)]
            total_ensemble_f1 += metric_max_over_ground_truths(compute_f1, ensemble_ans, literal_answer)
            total_ensemble_retrieval_recall += cur_avg_ensemble_retrieval_recall

        # sample using ppl
        min_ppl_ins = min(used_ins, key=lambda x: x['ppl'])
        min_ppl_index = used_ins.index(min_ppl_ins)
        min_ppl_ans = min_ppl_ins["final_answer"]

        if is_qa:
            if min_ppl_ans in label_dict[results["golds"][index]]:
                ppl_accurate_num += 1

        elif "popqa" in args.task:
            # specially handle this task, not qa, no retrieval performance
            if match(min_ppl_ins["final_answer"], results["golds"][index]):
                ppl_accurate_num += 1
        else:

            if not args.from_elastic_search:

                cur_ppl_retrieval_performance = min_ppl_ins["retrieval_performance"]
                total_ppl_retrieval_performance =  [cur + pre for cur, pre in zip(cur_ppl_retrieval_performance,
                                                                                 total_ppl_retrieval_performance)]
                cur_ppl_retrieval_recall = calculate_retrieval_recall(min_ppl_ins["retrieved_index"],
                                                                      gold_support_idxs[index])
            else:
                cur_ppl_retrieval_performance = [0, 0]
                total_ppl_retrieval_performance = [cur + pre for cur, pre in zip(cur_ppl_retrieval_performance,
                                                                                 total_ppl_retrieval_performance)]
                cur_ppl_retrieval_recall = calculate_retrieval_recall(min_ppl_ins["retrieved_index"],
                                                                      gold_support_idxs[index],
                                                                      is_title=True)

            total_ppl_f1 += metric_max_over_ground_truths(compute_f1, min_ppl_ins["final_answer"], literal_answer)


            total_ppl_retrieval_recall += cur_ppl_retrieval_recall

        # sample using confidence
        max_confidence_ins = max(used_ins, key=lambda x: x['confidence'])
        max_confidence_index = used_ins.index(max_confidence_ins)
        max_confidence_ans = max_confidence_ins["final_answer"]

        if is_qa:
            if max_confidence_ans in label_dict[results["golds"][index]]:
                confidence_accurate_num += 1
        elif "popqa" in args.task:
            # specially handle this task, not qa, no retrieval performance
            if match(max_confidence_ins["final_answer"], results["golds"][index]):
                confidence_accurate_num += 1
        else:

            if not args.from_elastic_search:
                cur_confidence_retrieval_performance = max_confidence_ins["retrieval_performance"]
                total_confidence_retrieval_performance = [cur + pre for cur, pre in zip(cur_confidence_retrieval_performance,
                                                                                 total_confidence_retrieval_performance)]

                cur_confidence_retrieval_recall = calculate_retrieval_recall(max_confidence_ins["retrieved_index"],
                                                                             gold_support_idxs[index])
            else:
                cur_confidence_retrieval_performance = [0, 0]
                total_confidence_retrieval_performance = [cur + pre for cur, pre in
                                                          zip(cur_confidence_retrieval_performance,
                                                              total_confidence_retrieval_performance)]

                cur_confidence_retrieval_recall = calculate_retrieval_recall(max_confidence_ins["retrieved_index"],
                                                                             gold_support_idxs[index],
                                                                             is_title=True)

            total_confidence_f1 += metric_max_over_ground_truths(compute_f1, max_confidence_ins["final_answer"], literal_answer)
            total_confidence_retrieval_recall += cur_confidence_retrieval_recall

        if is_qa or "popqa" in args.task:
            logger.info(f"\noracle accuracy: {oracle_accurate_num/ (index + 1)}"
                        f"\nensemble accuracy: {ensemble_accurate_num/ (index + 1)}"
                        f"\ncur ppl accuracy: {ppl_accurate_num / (index + 1)}"
                        f"\ncur confidence accuracy: {confidence_accurate_num / (index + 1)}")
        else:
            logger.info(f"\noracle f1: {total_max_f1 / (index + 1)}"
                        f"\nensemble f1: {total_ensemble_f1 / (index + 1)}"
                        f"\ncur ppl f1: {total_ppl_f1 / (index + 1)}"
                        f"\ncur confidence f1: {total_confidence_f1 / (index + 1)}")

            logger.info(
                f"\noracle retrieval_recall {total_max_retrieval_recall / (index + 1)} retrieval_performace {[cur / (index + 1) for cur in total_max_retrieval_performance]}"
                f"\nensemble retrieval_recall {total_ensemble_retrieval_recall / (index + 1)} retrieval_performace {[cur / (index + 1) for cur in total_ensemble_retrieval_performance]}"
                f"\nppl retrieval_recall {total_ppl_retrieval_recall / (index + 1)} retrieval_performace {[cur / (index + 1) for cur in total_ppl_retrieval_performance]}"
                f"\nconfidence retrieval_recall {total_confidence_retrieval_recall / (index + 1)} retrieval_performace {[cur / (index + 1) for cur in total_confidence_retrieval_performance]}")

    metric_mean = results["metric_mean"]

    if is_qa or "popqa" in args.task:
        logger.info(f"\noracle metric {oracle_accurate_num/len(results['prompts'])}"
                    f"\nensemble metric {ensemble_accurate_num / len(results['prompts'])}"
                    f"\nsample using ppl metric {ppl_accurate_num/len(results['prompts'])}"
                    f"\nsample using confidence metric {confidence_accurate_num / len(results['prompts'])}")

        results["oracle metric"] = oracle_accurate_num / len(results['prompts'])
        results["ensemble metric"] = ensemble_accurate_num / len(results['prompts'])
        results["ppl metric"] = ppl_accurate_num/len(results['prompts'])
        results["confidence metric"] = confidence_accurate_num / len(results['prompts'])
    else:
        logger.info(f"\noracle metric {total_max_f1 / len(results['prompts'])}"
                    f"\nensemble metric {total_ensemble_f1 / len(results['prompts'])}"
                    f"\nsample using ppl metric {total_ppl_f1 / len(results['prompts'])}"
                    f"\nsample using confidence metric {total_confidence_f1 / len(results['prompts'])}")

        logger.info(f"\noracle retrieval_recall {total_max_retrieval_recall / len(results['prompts'])} retrieval_performace {[cur/len(results['prompts']) for cur in total_max_retrieval_performance]}"
                    f"\nensemble retrieval_recall {total_ensemble_retrieval_recall / len(results['prompts'])} retrieval_performace {[cur / len(results['prompts']) for cur in total_ensemble_retrieval_performance]}"
                    f"\nppl retrieval_recall {total_ppl_retrieval_recall / len(results['prompts'])} retrieval_performace {[cur / len(results['prompts']) for cur in total_ppl_retrieval_performance]}"
                    f"\nconfidence retrieval_recall {total_confidence_retrieval_recall / len(results['prompts'])} retrieval_performace {[cur / len(results['prompts']) for cur in total_confidence_retrieval_performance]}")


        results["oracle metric"] = total_max_f1 / len(results['prompts'])
        results["ensemble metric"] = total_ensemble_f1 / len(results['prompts'])
        results["ppl metric"] = total_ppl_f1 / len(results['prompts'])
        results["confidence metric"] = total_confidence_f1 / len(results['prompts'])

        results["oracle retrieval_performace"] = [cur/len(results['prompts']) for cur in total_max_retrieval_performance]
        results["ensemble retrieval_performace"] = [cur/len(results['prompts']) for cur in total_ensemble_retrieval_performance]
        results["ppl retrieval_performace"] = [cur/len(results['prompts']) for cur in total_ppl_retrieval_performance]
        results["confidence retrieval_performace"] = [cur/len(results['prompts']) for cur in total_confidence_retrieval_performance]

        results["oracle retrieval_recall"] = total_max_retrieval_recall / len(results['prompts'])
        results["ensemble retrieval_recall"] = total_ensemble_retrieval_recall / len(results['prompts'])
        results["ppl retrieval_recall"] = total_ppl_retrieval_recall / len(results['prompts'])
        results["confidence retrieval_recall"] = total_confidence_retrieval_recall / len(results['prompts'])

    with open(os.path.join(args.run_name, f"sample_from_tree_calc_depth_{args.calc_depth}.json"), "w")as w:
        json.dump(results, w)
