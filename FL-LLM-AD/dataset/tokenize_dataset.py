import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dsname = 'hdfs'
model_name = 'HuggingFaceTB/SmolLM-360M'
dataset_path = f'../.dataset/{dsname}/train.csv'

data = pd.read_csv(dataset_path)
data = data[data['Label'] == 0]
# Prepare dataset for Hugging Face
data = data["Content"].tolist()
#data = [str(i) for i in data]
dataset_dic = {"text": data}

dataset = DatasetDict({
    "train": Dataset.from_dict({"text": data})
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.eos_token)

def preprocess_function(examples):
    #add eos token
    examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]
    ids = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024, padding_side="right")
    
    return ids

tokenized_datasets = dataset.map(preprocess_function, batched=True)

tokenized_datasets["train"] = tokenized_datasets["train"].map(lambda x: {"labels": x["input_ids"]}, batched=True)

#Save the tokenized dataset
tokenized_datasets.save_to_disk(f'../.dataset/{dsname}/tokenized')