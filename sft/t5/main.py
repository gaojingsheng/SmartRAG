from config import CFG
from trainer import CustomTrainer, compute_metrics
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from dataloader import QADataset, SubsetQADataset, CustomDataCollatorForSeq2Seq
import os

def get_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return tokenizer

if __name__ == "__main__":

    os.environ["WANDB_DISABLED"]="true"

    args = Seq2SeqTrainingArguments(
        "/flan-t5-base-warm-up/",
        evaluation_strategy = "epoch",
        learning_rate=0.0003, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        gradient_accumulation_steps=1, # 
    )

    model = get_model(CFG['model']['model_name'])   
    tokenizer = get_tokenizer(CFG['tokenizer']['tokenizer_name'])
    special_token_dict = {"additional_special_tokens": ["[Answer]", "[Search]"]}
    tokenizer.add_special_tokens(special_token_dict)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    tokenized_dataset = QADataset(json_path=CFG["data_path"], tokenizer=tokenizer)

    train_indices, val_indices = train_test_split(
        list(range(len(tokenized_dataset))),  
        test_size=0.01,  
        random_state=42  
    )

    tokenized_train_dataset = SubsetQADataset(tokenized_dataset, train_indices)
    tokenized_val_dataset = SubsetQADataset(tokenized_dataset, val_indices)

    data_collator = CustomDataCollatorForSeq2Seq(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate(eval_dataset=tokenized_val_dataset)

