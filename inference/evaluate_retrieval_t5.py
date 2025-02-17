import numpy as np
import string
import re
from collections import Counter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
import requests
import os
import time
from argparse import ArgumentParser
import random 
import copy

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

retrieval_tokens_names = ["[Answer]", "[Search]"]

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



def load_model_tokenizer_for_inference(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model,tokenizer

def generate_response(prompt_text, model, tokenizer, max_length=150, num_return_sequences=1):
    model.eval()

    tokenized_input = tokenizer(prompt_text, return_tensors="pt")

    summary_ids = model.generate(
        input_ids= tokenized_input["input_ids"].cuda(),
        max_length=max_length,  
        num_beams=4,     
        no_repeat_ngram_size=3,  
        early_stopping=True,
        num_return_sequences=num_return_sequences,
        )

    responses = []
    for response_id in summary_ids :
        response = tokenizer.decode(response_id)
        responses.append(response)

    return responses[0].replace("<pad>","").replace("</s>","")

def generate_response_with_prob(prompt_text, model, tokenizer, thres_hold, max_length=150, num_return_sequences=1):
    model.eval()
    tokenized_input = tokenizer(prompt_text, return_tensors="pt")
    
    input_ids = tokenized_input["input_ids"].cuda()

    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]]).cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    first_token_logits = outputs.logits[:, 0, :]

    first_token_probs = torch.softmax(first_token_logits, dim=-1)

    token_probs_answer = first_token_probs[0, 32101]
    token_probs_retrieval = first_token_probs[0, 32100]
    print(token_probs_answer, token_probs_retrieval)


    if token_probs_retrieval > thres_hold:
        model.generation_config.decoder_start_token_id = 32100
    else:
        model.generation_config.decoder_start_token_id = 32101

    summary_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,  
        num_beams=4,     
        no_repeat_ngram_size=3,  
        early_stopping=True,
        num_return_sequences=num_return_sequences,
    )

    responses = []
    for response_id in summary_ids:
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return model.generation_config.decoder_start_token_id, responses[0].replace("<pad>","").replace("</s>","")


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

def get_first_word_and_remaining(sentence):
    parts = sentence.split(maxsplit=1)
    first_word = parts[0]
    remaining_sentence = parts[1] if len(parts) > 1 else ""
    return first_word, remaining_sentence.strip()

def hits(ans, res, dn, dl=False):
    assert type(ans) == list, print("answer tupe is not list")
    n = 0
    res = normalize_answer(res)
    hit_ = []
    for a in ans:
        a = normalize_answer(a)
        hit_.append(res.count(a))

    for i in range(sum(hit_)):
        n += dn ** (i)
    if dl:
        n = n / len(res.split())
    print(hit_)
    print(n)
    return n

if __name__ == "__main__":

    parser = ArgumentParser(description="Evaluate the results of retrieval LMs")
    parser.add_argument("--base_model_path", type=str, help="path to the base model path file", default="")
    parser.add_argument("--save_evaluate_path", type=str, help="path to the save path file", default="")

    parser.add_argument(
        "--checkpoint", type=str, help="trained ppo checkpoint path", default=""
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="evaluate datasets",
        default="ambignq",
    )
    parser.add_argument(
        "--sample_test", action="store_true", help="Whether to mini dataset to test"
    )
    parser.add_argument("--threshold", type=float, help="retrieval threshold", default=0.5)
    
    args = parser.parse_args()

    base_model_path = args.base_model_path
    print(base_model_path)
    if args.dataset == "popqa":
        test_data_path = "SmartRAG/datasets/tasks/popqa/test.jsonl"
        args.save_evaluate_path += "popqa_"
        test_data = load_jsonl(test_data_path)

    elif args.dataset == "ambignq":
        test_data_path = "SmartRAG/datasets/tasks/ambignq/dev.jsonl"
        args.save_evaluate_path += "ambigqa_"
        test_data = load_jsonl(test_data_path)

    elif args.dataset == "hotpotqa":
        test_data_path = "SmartRAG/datasets/tasks/hotpotqa/hotpot_dev_v1_simplified.json"
        args.save_evaluate_path += "hotpotqa_"
        test_data = load_json(test_data_path)
    else:
        raise ValueError("Dataset name is wrong")
    
    if args.sample_test:
        test_data = test_data[:200]
        args.save_evaluate_path += "200_"

    args.save_evaluate_path += str(args.threshold)
    args.save_evaluate_path += "_"
    
    model, tokenizer = load_model_tokenizer_for_inference(base_model_path)
    model.cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
        model.load_state_dict(checkpoint["policy_state"]["policy_model"]) 

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

        model.generation_config.decoder_start_token_id = 0

        first_token, predicted = generate_response_with_prob(question_input, model, tokenizer, args.threshold)

        if first_token == 32101:
            predict_list.append(predicted)
            answers_list.append(answers)
            temp_data["predict"] = predicted

        elif first_token == 32100:
            try:
                retrieve_text, search_time = bing(predicted)
            except:
                retrieve_text = ""
            temp_data["Search"] = predicted
            temp_data["retrieve_text"] = retrieve_text

            if hits(answers, retrieve_text, dn=0, dl=False) != 0:
                hit_num += 1
            else:
                no_hit_num += 1

            retrival_input = RETRIEVAL_INSTRUCTION.format(input=temp_data["question"], search=retrieve_text)

            model.generation_config.decoder_start_token_id = 0 
            predicted = generate_response(retrival_input, model, tokenizer)
            answer_query, remaining_sentence = get_first_word_and_remaining(predicted)

            temp_data["predict"] = predicted
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
