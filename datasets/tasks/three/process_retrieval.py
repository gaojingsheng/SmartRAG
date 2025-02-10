import json
import random

if __name__ == "__main__":
    input_path = "/mnt/workspace/user/gaojingsheng/LLM/retrieval/RL4LMs/datasets/tasks/three/dev.json"
    output_path = "/mnt/workspace/user/gaojingsheng/LLM/retrieval/RL4LMs/datasets/tasks/three/dev.json"
    
    # Load the JSON data
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # Modify the data by adding a "retrieval" key with a random boolean value
    for item in data:
        item["retrieval"] = random.choice([True, False])
    
    # Save the modified data to a new JSON file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Modified data has been saved to {output_path}")
