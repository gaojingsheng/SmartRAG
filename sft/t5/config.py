'''
This is the configuration file for the project.
if you want to change any of the parameters, you can change it here. We will change the model and tokenizer in this file.
'''

CFG = {
    'data_path': 'SmartRAG/rl/datasets/warmup/warm_up_t5.json',
    'train_size': 0.7,
    'val_size': 0.1,
    'test_size': 0.2,

    'tokenizer': {
        'tokenizer_type': 'AutoTokenizer',
        'tokenizer_name': 'flan-t5-large',
    },
    'model': {
        'model_type': 'AutoModelForSeq2SeqLM',
        'model_name': 'flan-t5-large'
    },
    'inference': {
        'inferece_task': 'Question Answer',
        'model_path': 'weights'
    }
    }
