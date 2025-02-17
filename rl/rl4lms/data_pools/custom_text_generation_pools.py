from rl4lms.data_pools.text_generation_pool import TextGenPool, Sample
from rl4lms.data_pools.task_utils.totto import preprocess_utils
from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import os
from urllib.request import urlretrieve
from pathlib import Path
import pandas
from collections import defaultdict
import zipfile
import json


class Ambig(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_prefix: str = "", ifdebug: bool = False):
        split_ = split
        split = Ambig.gen_split_name(split_)
        infile = f'/mnt/workspace/user/gaojingsheng/LLM/retrieval/RAG-query-rewriting/datasets/tasks/ambignq/{split}.jsonl'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        if split_ == "val":
            lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]
        if type(lines[0]['answer']) == str: #  answer -> list type
            for l in lines:
                l['answer'] = [l['answer']]
        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"],
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "dev"
        else:
            raise NotImplementedError
        return split_name

class naq(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "", prompt_prefix: str = "", ifdebug: bool = False):

        split_ = split
        split = popqa.gen_split_name(split_)
        infile = f'RL4LMs/datasets/tasks/naq/{split}.json'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        # if split_ == "val":
        #     lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]

        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"] + prompt_suffix,
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class three(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "", prompt_prefix: str = "", ifdebug: bool = False):

        split_ = split
        split = popqa.gen_split_name(split_)
        if split == "train":
            split = "train_retrieval"
        infile = f'RL4LMs/datasets/tasks/three/{split}.json'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))

        if ifdebug:
            lines = lines[:10]

        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []

        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"] + prompt_suffix,
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

def options2choices(options):
    choices = ""
    for item in options:
        
        choices += item
        choices += ": "
        choices += options[item]
        choices += " "
    return "\n\n" + choices
    
class SelectionThree(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "", prompt_prefix: str = "", ifdebug: bool = False):

        split_ = split
        split = popqa.gen_split_name(split_)

        infile = f'RL4LMs/datasets/tasks/SelectionThree/{split}.json'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))

        if ifdebug:
            lines = lines[:10]

        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []

        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"] + options2choices(item["option"]) + prompt_suffix,
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class ambignq(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "", prompt_prefix: str = "", ifdebug: bool = False):

        split_ = split
        split = popqa.gen_split_name(split_)
        infile = f'RL4LMs/datasets/tasks/ambignq/{split}.jsonl'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        # if split_ == "val":
        #     lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]

        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"] + prompt_suffix,
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class popqa(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "", prompt_prefix: str = "", ifdebug: bool = False):

        split_ = split
        split = popqa.gen_split_name(split_)
        infile = f'RL4LMs/datasets/tasks/popqa/{split}.jsonl'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        # if split_ == "val":
        #     lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]

        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"] + prompt_suffix,
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class triviaqa(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "", prompt_prefix: str = "", ifdebug: bool = False):

        split_ = split
        split = popqa.gen_split_name(split_)
        infile = f'RL4LMs/datasets/tasks/triviaqa/{split}.json'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        # if split_ == "val":
        #     lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]

        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"] + prompt_suffix,
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class moviedata(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "", prompt_prefix: str = "", ifdebug: bool = False):

        split_ = split
        split = popqa.gen_split_name(split_)
        infile = f'RL4LMs/datasets/tasks/moviedata/{split}.json'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))

        if ifdebug:
            lines = lines[:10]

        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"] + prompt_suffix,
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class mmlusocial(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_prefix: str = "", ifdebug: bool = False):
        split_ = split
        split = mmlusocial.gen_split_name(split_)
        infile = f'/mnt/workspace/user/gaojingsheng/LLM/retrieval/RAG-query-rewriting/datasets/tasks/mmlu/test/mysplit/socialsciences-{split}-add.jsonl'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        # lines = lines[:10000]
        # if split_ == "val":
        #     lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]
        if type(lines[0]['answer']) == str: #  answer -> list type
            for l in lines:
                l['answer'] = [l['answer']]
        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"],
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class mmluhumanities(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_prefix: str = "", ifdebug: bool = False):
        split_ = split
        split = mmluhumanities.gen_split_name(split_)
        infile = f'/mnt/workspace/user/gaojingsheng/LLM/retrieval/RAG-query-rewriting/datasets/tasks/mmlu/test/mysplit/humanities-{split}-add.jsonl'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        # lines = lines[:10000]
        # if split_ == "val":
        #     lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]
        if type(lines[0]['answer']) == str: #  answer -> list type
            for l in lines:
                l['answer'] = [l['answer']]
        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"],
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class mmluother(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_prefix: str = "", ifdebug: bool = False):
        split_ = split
        split = mmluother.gen_split_name(split_)
        infile = f'/mnt/workspace/user/gaojingsheng/LLM/retrieval/RAG-query-rewriting/datasets/tasks/mmlu/test/mysplit/other-{split}-add.jsonl'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        # lines = lines[:10000]
        # if split_ == "val":
        #     lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]
        if type(lines[0]['answer']) == str: #  answer -> list type
            for l in lines:
                l['answer'] = [l['answer']]
        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"],
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

class mmlustem(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_prefix: str = "", ifdebug: bool = False):
        split_ = split
        split = mmlustem.gen_split_name(split_)
        infile = f'/mnt/workspace/user/gaojingsheng/LLM/retrieval/RAG-query-rewriting/datasets/tasks/mmlu/test/mysplit/stem-{split}-add.jsonl'
        if infile.split(".")[-1] == 'jsonl':
            lines = open(infile, 'r', encoding='utf8').readlines()
            lines = [json.loads(l) for l in lines] 
        elif infile.split(".")[-1] == 'json':
            lines = json.load(open(infile, 'r', encoding='utf8'))
        # lines = lines[:10000]
        # if split_ == "val":
        #     lines = lines[:1000]
        if ifdebug:
            lines = lines[:10]
        if type(lines[0]['answer']) == str: #  answer -> list type
            for l in lines:
                l['answer'] = [l['answer']]
        print(f"load {str(len(lines))} {split} examples.")
        print('eg: ', lines[0])
        samples = []
        for ix, item in enumerate(lines):
            sample = Sample(id=f"{split}_{ix}",
                           prompt_or_input_text=prompt_prefix + item["question"],
                           references=[item["answer"]]
                           )
            samples.append(sample)
        print(f"sample {str(len(lines))} {split} examples.")
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "test" or "val":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name

def download_file_using_url(url: str, dest_path: str):
    urlretrieve(url, dest_path)

class DailyDialog(TextGenPool):
    EOU_TOKEN = "<EOU>"
    @classmethod
    def prepare(cls, split: str, context_size: int):
        split = CommonGen.gen_split_name(split)
        dataset = load_dataset("daily_dialog", split=split)
        samples = []
        utterance_id = 0
        for item in dataset:
            contexts = []
            for utterance, emotion, intent in zip(item["dialog"],
                                                  item["emotion"],
                                                  item["act"]):
                if len(contexts) >= context_size:
                    context = DailyDialog.EOU_TOKEN.join(contexts[-context_size:]) 
                    context += " " + DailyDialog.EOU_TOKEN
                    target = utterance + DailyDialog.EOU_TOKEN
                    sample = Sample(id=utterance_id, 
                                    prompt_or_input_text=context, 
                                    references=[target],
                                    meta_data={
                                        "emotion": [emotion],
                                        "intent": [intent]
                                    })
                    samples.append(sample)
                contexts.append(utterance)
                utterance_id += 1

        dp_instance = cls(samples)
        return dp_instance
