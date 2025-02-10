import json

# 解析JSON数据
data = json.load(open("/mnt/workspace/user/gaojingsheng/LLM/retrieval/RL4LMs/datasets/tasks/three/train.json", "rb"))

# 提取answers并将整个结构存储到新的列表中
processed_data = []
for item in data:
    processed_item = {
        "question": item['question'],
        "answer": [item['answer']]
    }
    processed_data.append(processed_item)

# 输出结果
# print(json.dumps(processed_data, indent=4))

# 存储为JSON文件
with open("/mnt/workspace/user/gaojingsheng/LLM/retrieval/RL4LMs/datasets/tasks/three/train.json", 'w') as file:
    json.dump(processed_data, file, indent=4)

# 打印存储成功信息
print("Processed data has been successfully stored in 'processed_data.json'.")


