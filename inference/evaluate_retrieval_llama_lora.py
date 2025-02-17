import numpy as np
import string
import re
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
import requests
import os
import time
from argparse import ArgumentParser
import random 
import copy
from peft import set_peft_model_state_dict, PeftModel

ORIGIN_INSTRUCTION = """You will be presented with a question. If you know the answer, please respond directly. If you don't know the answer, use the Bing search engine to find the necessary information and then answer the question based on your observation.

Question: {input}

Please format your output as follows:

1. If you choose to answer the question directly, please use: "[Answer] YOUR_ANSWER"
2. If you choose to use the Bing search engine, please use: "[Search] YOUR_SEARCH_QUERY"

Please output:
"""

RETRIEVAL_INSTRUCTION = """You will be presented with a question. If you know the answer, please respond directly. If you don't know the answer, use the Bing search engine to find the necessary information and then answer the question based on your observation. 

Question: {input}

Observation: {search} 

Please format your output as follows:

1. If you choose to answer the question directly, please use: "[Answer] YOUR_ANSWER"
2. If you choose to use the Bing search engine, please use: "[Search] YOUR_SEARCH_QUERY"

Please output:
"""

URL_BING = "https://www.bingapis.com/api/v7/search?q={query}&appid=yourapi&count=4"


def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    for i in range(len(data)):
        data[i]["answer"] = [data[i]["answer"]]
    return data

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            single_data = json.loads(line)
            data.append(single_data)
    return data

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    return 

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1
    return 100 * (match_count / len(preds))

def accuracy_list_total(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        if type(label) == str:
            if exact_match_score(pred, label):
                match_count += 1
        else:
            for target in label:
                if exact_match_score(pred, target):
                    match_count += 1
                    break
    print("match_count is: ", match_count)
    print("len(preds) is: ", len(preds))
    return 100 * (match_count / len(preds))

def f1(decoded_preds, decoded_labels):
    f1_all = []
    for prediction, answers in zip(decoded_preds, decoded_labels):
        if type(answers) == list:
            if len(answers) == 0:
                return 0
            f1_all.append(np.max([qa_f1_score(prediction, gt)
                          for gt in answers]))
        else:
            f1_all.append(qa_f1_score(prediction, answers))
    return 100 * np.mean(f1_all)

def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def bing(topic):
    url_bing = URL_BING.format(query=topic)
    old_time = time.time()
    response = requests.get(url_bing, timeout=3)
    response_json = response.json()
    search_time = time.time() - old_time
    res_text = []
    web_pages = response_json.get('webPages', {}).get('value', [])

    for page  in web_pages:

        try:
            single_data = page['name'] + ": " + page['snippet']
            
            res_text.append(single_data)
            try:
                for link in page.get('deepLinks', []):
                    single_data = link['name'] + ": " + link['snippet']
                    res_text.append(single_data)
            except:
                pass
        except:
            pass
    obs= " ".join(res_text)

    return obs, search_time

def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0

def load_model_tokenizer_for_inference(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(prompt, model, tokenizer):

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4, 
        early_stopping=True 
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

def get_first_word_and_remaining(sentence):
    parts = sentence.split(maxsplit=1)
    first_word = parts[0]
    remaining_sentence = parts[1] if len(parts) > 1 else ""
    return first_word, remaining_sentence.strip()

def extract_answer(prediction):
    answer_start = "Please output:"
    start_idx = prediction.find(answer_start) + len(answer_start)
    return prediction[start_idx:].strip()


def hits(ans, res, dn, dl=False):
    assert type(ans) == list, "answer type is not list"
    n = 0
    res = normalize_answer(res)
    hit_ = [res.count(normalize_answer(a)) for a in ans]
    for i in range(sum(hit_)):
        n += dn ** (i)
    if dl:
        n /= len(res.split())
    print(hit_)
    print(n)
    return n

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate the results of retrieval LMs")
    parser.add_argument("--base_model_path", type=str, help="path to the base model path file", default="checkpoint/llama2-7b-chat-hf")
    parser.add_argument("--lora_path", type=str, help="path to the base model path file", default="self-rag/data_creation/critic/sft_llama_lora_no_special/checkpoint-900")
    parser.add_argument("--save_evaluate_path", type=str, help="path to the save path file", default="/cpfs/user/gaojingsheng/JapanA100/ICLR/llama7b/num/test_llama7b_sft_retrieval_")
    parser.add_argument('--gen_mode', choices=['normal', 'answer', 'retrieval'], 
                    default='normal', help="Generation mode, choose from 'normal', 'answer', 'retrieval'. Default is 'normal'")
    parser.add_argument("--checkpoint", type=str, help="trained ppo checkpoint path", default="")
    parser.add_argument("--dataset", type=str, help="evaluate datasets", default="ambignq")
    parser.add_argument(
        "--not_sample_test", action="store_true", help="Whether to mini dataset to test"
    )
    args = parser.parse_args()

    if args.dataset == "popqa":
        test_data_path = "/cpfs/user/gaojingsheng/JapanA100/RL4LMs/datasets/tasks/popqa/test.jsonl"
        args.save_evaluate_path += "popqa_"
        test_data = load_jsonl(test_data_path)
    elif args.dataset == "ambignq":
        test_data_path = "/cpfs/user/gaojingsheng/JapanA100/RL4LMs/datasets/tasks/ambignq/test.jsonl"
        args.save_evaluate_path += "ambigqa_"
        test_data = load_jsonl(test_data_path)
    elif args.dataset == "hotpotqa":
        test_data_path = "/cpfs/user/gaojingsheng/JapanA100/RL4LMs/datasets/tasks/hotpotqa/hotpot_dev_v1_simplified.json"
        args.save_evaluate_path += "hotpotqa_"
        test_data = load_json(test_data_path)
    else:
        raise ValueError("Dataset name is wrong")
    
    random.seed(10)
    random.shuffle(test_data)

    if not args.not_sample_test:
        test_data = test_data[:1000]
        args.save_evaluate_path += "1000_"
    model, tokenizer = load_model_tokenizer_for_inference(args.base_model_path)

    model = PeftModel.from_pretrained(model, args.lora_path)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")

        model = set_peft_model_state_dict(model, checkpoint["policy_state"]["policy_model"])

        filename_with_extension = os.path.basename(args.checkpoint)
        filename_without_extension, extension = os.path.splitext(filename_with_extension)
        args.save_evaluate_path += filename_without_extension

    predict_list = []
    answers_list = []
    target_save_list = []
    hit_num = 0 
    no_hit_num = 0 
    for single_data in tqdm(test_data):
        temp_data = copy.deepcopy(single_data)
        question_input = ORIGIN_INSTRUCTION.format(input=temp_data["question"])
        answers = temp_data["answer"]

        if args.gen_mode == "answer":
            question_input += "[Answer]"
        elif args.gen_mode == "retrieval":
            question_input += "[Search]"
        else:
            pass

        predicted = extract_answer(generate_response(question_input, model, tokenizer))
        answer_query, remaining_sentence = get_first_word_and_remaining(predicted)

        if answer_query == "[Answer]":
            predict_list.append(remaining_sentence)
            answers_list.append(answers)
            temp_data["predict"] = remaining_sentence
        elif answer_query == "[Search]":
            try:
                retrieve_text, search_time = bing(remaining_sentence)

            except:
                retrieve_text = ""
            temp_data["Search"] = remaining_sentence
            temp_data["retrieve_text"] = retrieve_text

            if hits(answers, retrieve_text, dn=0, dl=False) != 0:
                hit_num += 1
            else:
                no_hit_num += 1


            retrival_input = RETRIEVAL_INSTRUCTION.format(input=temp_data["question"], search=retrieve_text)

            predicted = extract_answer(generate_response(retrival_input, model, tokenizer))
            answer_query, remaining_sentence = get_first_word_and_remaining(predicted)

            temp_data["predict"] = remaining_sentence
            predict_list.append(remaining_sentence)
            answers_list.append(answers)
        else:
            print("format error")
            print(predicted)
            temp_data["predict"] = "format error"
        target_save_list.append(temp_data)

    accuracy = round(accuracy_list_total(predict_list, answers_list), 3)
    f1_score = round(f1(predict_list, answers_list), 3)
    args.save_evaluate_path += str(accuracy)
    args.save_evaluate_path += ("_f1_" + str(f1_score) + ".json")

    print("The whole accuracy is: ", accuracy)
    print("The whole f1_score is: ", f1_score)
    print("hit_num is: ", hit_num)
    print("no_hit_num is: ", no_hit_num)

    save_json(target_save_list, args.save_evaluate_path)
