import json
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from torch.nn.utils.rnn import pad_sequence
import random 

class QADataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f) # [:100]
        random.shuffle(self.data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        item = self.data[idx]

        model_inputs = self.tokenizer.encode_plus(item['input'], 
                                    max_length=self.max_length, 
                                    padding='max_length', 
                                    truncation=True, 
                                    return_tensors="pt")
                
        labels = self.tokenizer.encode_plus(item['output'], 
                                    max_length=self.max_length, 
                                    padding='max_length', 
                                    truncation=True, 
                                    return_tensors="pt")
        
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

class SubsetQADataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model=None, label_pad_token_id=0, padding=True):
        super().__init__(tokenizer, model, label_pad_token_id, padding)
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.padding = padding

    def __call__(self, features):
        
        labels = [feature['labels'] for feature in features]
        labels = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)
        
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt"
        )

        batch["labels"] = labels.squeeze()  
        batch["input_ids"] = batch["input_ids"].squeeze()  
        batch["attention_mask"] = batch["attention_mask"].squeeze() 
        return batch
