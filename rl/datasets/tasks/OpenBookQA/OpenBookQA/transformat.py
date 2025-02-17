import json
import os

base_path = '/Users/jike/PycharmProjects/self-improvement/data/OpenBookQA/OpenBookQA-V1-Sep2018/Data/Main'

tgt_path = '/Users/jike/PycharmProjects/self-improvement/data/OpenBookQA/formatted'


def transformat_eval(mode):
    datas = []
    dataset = {}
    new_datas = []
    with open(os.path.join(base_path, f"{mode}.jsonl")) as fp:
        lines = list(fp)
        for line in lines:
            datas.append(json.loads(line))
    for data in datas:
        new_data = dict()
        new_data['question'] = data['question']['stem']
        new_data['answer'] = data['answerKey']
        new_data['option'] = dict({i['label']: i['text'] for i in data['question']['choices']})
        new_datas.append(new_data)
    dataset['OpenBookQA'] = new_datas
    with open(os.path.join(tgt_path, f"{mode}.json"), 'w') as fp:
        json.dump(dataset, fp, ensure_ascii=False, indent=2)


def transformat_for_sft(mode):
    datas = []
    dataset = {}
    new_datas = []
    with open(os.path.join(base_path, f"{mode}.jsonl")) as fp:
        lines = list(fp)
        for line in lines:
            datas.append(json.loads(line))
    for data in datas:
        new_data = dict()
        new_data['question'] = data['question']['stem']
        new_data['answer_idx'] = data['answerKey']

        new_data['options'] = dict({i['label']: i['text'] for i in data['question']['choices']})
        new_data['answer'] = new_data['options'][data['answerKey']]
        new_datas.append(json.dumps(new_data, ensure_ascii=False))
    tgt_datas = "\n".join(new_datas)
    with open(os.path.join(tgt_path, f"{mode}.jsonl"), 'w') as fp:
        fp.write(tgt_datas)


# splits = ['train', 'dev', 'test']
# for split in splits:
#     transformat_eval(split)

splits = ['train']
for split in splits:
    transformat_for_sft(split)


# data_path = 'data/OpenBookQA/OpenBookQA-V1-Sep2018/Data/Main/openbook.txt'
# output_path = 'data/OpenBookQA/openbook_book_cn.json'
#
# with open(data_path) as fp:
#     lines = [line.strip().replace("\"", "") for line in list(fp)]
#
# datas = []
# for line in lines:
#     datas.append({"text": line})
#
# with open(output_path, 'w') as fp:
#     json.dump(datas, fp, ensure_ascii=False, indent=2)
