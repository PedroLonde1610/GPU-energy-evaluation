import os
import random
from utils import initialize_model, load_dataset, split_data, train_client, aggregate_models, set_adapters, save_global_model, get_adapters, cosine_learning_rate
import warnings
import torch
import hashlib
import argparse
import numpy as np
from datasets import Dataset
from codecarbon import EmissionsTracker


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Federated Learning Simulation")
parser.add_argument('--sim_name', type=str, required=True, help='Name of the simulation')
parser.add_argument('--num_rounds', type=int, default=50, help='Number of training rounds')
parser.add_argument('--num_clients', type=int, default=50, help='Total number of clients')
parser.add_argument('--client_frac', type=float, default=0.1, help='Fraction of clients selected per round')
parser.add_argument('--model_name', type=str, default="HuggingFaceTB/SmolLM-135M", help='Pre-trained model name or path')
parser.add_argument('--dataset_path', type=str,  default='../.dataset/hdfs/tokenized', help='Path to the dataset')
parser.add_argument('--lora', action='store_true', help='Enable LoRA (Low-Rank Adaptation)')
parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
parser.add_argument('--nrows', type=int, default=None, help='Number of rows to load from the dataset (for debugging)')
parser.add_argument('--cuda_device', type=str, default="0", help='CUDA device to use')
parser.add_argument('--max_steps', type=int, default=10, help='Maximum training steps per client')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for client training')

args = parser.parse_args()

tracker = EmissionsTracker(
    project_name="federated_fl",
    output_dir=f"./fl-results/{args.sim_name}/emissions",
    output_file="emissions.csv"
)
tracker.start()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

# Initialize variables
SIM_NAME = args.sim_name
NUM_ROUNDS = args.num_rounds
NUM_CLIENTS = args.num_clients
CLIENT_FRAC = args.client_frac
MODEL_NAME = args.model_name
path = args.dataset_path
lora = args.lora
lora_rank = args.lora_rank
nrows = args.nrows
max_steps = args.max_steps
batch_size = args.batch_size

global_model, tokenizer = initialize_model(MODEL_NAME, lora_rank=lora_rank, sim_name=SIM_NAME, lora = lora) 
rs = random.SystemRandom()

tokenized_datasets = load_dataset(path, nrows=nrows)
split_data(tokenized_datasets, NUM_CLIENTS, SIM_NAME)

for round in range(1, NUM_ROUNDS+1):

    #Select Clients
    clients = rs.sample(list(range(NUM_CLIENTS)), int(NUM_CLIENTS*CLIENT_FRAC))
    #clients = [0]
    
    print(f"Exp: {SIM_NAME} - Round {round}: Clients Selected {clients}")

    # Train clients
    clients_models = []
    for client in clients:
        new_lr = cosine_learning_rate(current_round = round, total_rounds = NUM_ROUNDS, initial_lr=1e-3, min_lr=1e-5)
        print(f"Round {round}: Training Client {client} with lr {new_lr}")

        client_dataset = Dataset.load_from_disk(f"./fl-results/{SIM_NAME}/round_0/client_{client}")
        client_model = train_client(
            int(client), client_dataset, round, SIM_NAME, tokenizer,
            max_steps=max_steps, lr=new_lr, batch_size=batch_size, model_name=MODEL_NAME,
            lora=lora
        )
        
        clients_models.append(client_model)
        print(f"Round {round}: Client {client} trained")

    # Aggregate model
    aggregated_adapters = aggregate_models(clients_models, lora = lora)
    if lora:
        set_adapters(global_model, aggregated_adapters)
        save_global_model(global_model, round, SIM_NAME)
    else:
        save_global_model(aggregated_adapters, round, SIM_NAME)

tracker.stop()
