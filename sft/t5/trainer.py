
from transformers import AutoTokenizer, Seq2SeqTrainer
import torch.nn as nn
from config import CFG
from datasets import load_metric
import nltk
import numpy as np

class CustomTrainer(Seq2SeqTrainer):


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Assuming you have a sequence-to-sequence model, you need to compute the loss differently
        # You can use a loss function like nn.CrossEntropyLoss, but applied to each timestep and averaged
        loss_fct = nn.CrossEntropyLoss(reduction="mean")

        # Flatten the logits and labels to make them compatible with the loss function
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # Compute the loss
        loss = loss_fct(logits_flat, labels_flat)
        # loss = calculate_sparse_categorical_cross_entropy_loss(logits, labels)
        # return (loss,outputs)
        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(eval_pred):
    return {}

def compute_metrics_v1(eval_pred):

    metric = load_metric("rouge")
    tokenizer = AutoTokenizer.from_pretrained((CFG['tokenizer']['tokenizer_name']), use_fast=True)

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)

    score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    result_dict = {}
    for rouge_type in score_keys:
        rouge_score = result[rouge_type].mid.fmeasure
        result_dict[f"lexical/rouge_{rouge_type}"] = rouge_score

    # Extract a few results
    # result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result_dict["gen_len"] = np.mean(prediction_lens)

    print(result_dict)
    return result_dict # {k: round(v, 4) for k, v in result.items()}


