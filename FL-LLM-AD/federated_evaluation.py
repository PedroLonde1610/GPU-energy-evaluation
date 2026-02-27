import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from peft import PeftModel
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch
import argparse
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Evaluate Next-Token Prediction in Federated Learning")
parser.add_argument('--sim_name', type=str, required=True, help='Simulation name')
parser.add_argument('--model_name', type=str, default="HuggingFaceTB/SmolLM-135M", help='Pre-trained model name or path')
parser.add_argument('--dataset_path', type=str, default='../.dataset/hdfs/test.csv', help='Path to the dataset CSV')
parser.add_argument('--nrows', type=int, default=None, help='Number of rows to load from the dataset (for debugging)')
parser.add_argument('--n_rounds', type=int, default=50, help='Number of rounds to evaluate')
parser.add_argument('--lora', action='store_true', help='Enable LoRA (Low-Rank Adaptation)')
parser.add_argument('--cuda_device', type=str, default="0", help='CUDA device to use')
args = parser.parse_args()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

# Configuration
SIM_NAME = args.sim_name
model_name = args.model_name
dataset_path = args.dataset_path
nrows = args.nrows
N_ROUNDS = args.n_rounds
lora = args.lora

RES_NAME = f"{SIM_NAME}_complete"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_in_top_k(top_k, target):
    #implement in efficient way
    return target in top_k


def next_token_top_k(data, model, tokenizer,):

    """
    Predict the next token for a given text using a llm model.
    Calculate with the correct token is in the top k predictions.
    """
    accuracies = {'top1':[], 'top3':[], 'top5':[], 'top10':[]}

    model.to(device)
    with torch.no_grad():
        for text in data["text"]:
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024, padding_side="right")
            inputs = {key: inputs[key].to(device) for key in inputs}

            outputs = model(**inputs)
            logits = outputs.logits

            for k in [1, 3, 5, 10]:
                top_k = torch.topk(logits, k, dim=-1).indices
                top_k = top_k.cpu().numpy() 
                
                tokens = inputs['input_ids'][0].cpu().numpy()
                correct = sum(is_in_top_k(top_k[0], token) for token in tokens)

                accuracies[f'top{k}'].append(correct/len(tokens))
        
    return accuracies

if __name__ == '__main__':

    data = pd.read_csv(dataset_path, nrows=nrows)
    #shuffle data
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)

    #get 50/50 of each class
    data = pd.concat([data[data['Label'] == 1].head(1000), data[data['Label'] == 0].head(1000)])

    print(len(data[data['Label'] == 1]), len(data[data['Label'] == 0]))

    labels = data["Label"].tolist()
    data = data["Content"].tolist()
    data = [str(i) for i in data]
    dataset_dic = {"text": data}
    df_acc = pd.DataFrame()
    df_acc.to_csv(f"results_accs_{RES_NAME}.csv", index=False)

    df_f1 = pd.DataFrame()
    df_f1.to_csv(f"results_f1_{RES_NAME}.csv", index=False)

    for round in range(1, N_ROUNDS+1):
        
        if round == 0:
            round = 1

        lora_path = f"fl-results/{SIM_NAME}/round_{round}/global_model"

        
        model_pretrained = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        if lora:
            model_ft = PeftModel.from_pretrained(model_pretrained, lora_path)
            model_ft.to(device)
        else:
            model_ft = AutoModelForCausalLM.from_pretrained(lora_path)

        accuracies = next_token_top_k(dataset_dic, model_ft, tokenizer)
        for k in [1, 3, 5, 10]:

            df_results = pd.DataFrame(accuracies)
            df_results['label'] = labels
            df_results['round'] = round 
            df_results['k'] = k
        
            df_acc = pd.concat([df_acc, df_results])

            df_acc.to_csv(f"results_accs_{RES_NAME}.csv", index=False)

            ths = np.linspace(0, 1, 1000)

            best_f1 = 0
            best_th = 0

            for th in ths:
                df_results['pred'] = df_results[f'top{k}'] < th
                df_results['pred'] = df_results['pred'].astype(int)

                f1 = f1_score(df_results['label'], df_results['pred'])
                precision = precision_score(df_results['label'], df_results['pred'])
                recall = recall_score(df_results['label'], df_results['pred'])

                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th

            df_results['pred'] = df_results[f'top{k}'] < best_th
            df_results['pred'] = df_results['pred'].astype(int)

            f1 = f1_score(df_results['label'], df_results['pred'])
            precision = precision_score(df_results['label'], df_results['pred'])
            recall = recall_score(df_results['label'], df_results['pred'])

            print(f'Round: {round}')
            print(f'K: {k}')
            print(f'Threshold: {best_th}')
            print(f'F1: {f1}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')

            df_f1 = pd.concat([df_f1, pd.DataFrame({"round": [round], "k": [k], "threshold": [best_th], "f1": [f1], "precision": [precision], "recall": [recall]})])
            df_f1.to_csv(f"results_f1_{RES_NAME}.csv", index=False)