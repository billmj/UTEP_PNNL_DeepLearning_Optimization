import os
import time
import torch
import torch.nn as nn
import pandas as pd
import argparse
import signal
import sys
import random
from utils import FLconfig, preprocess_dataset, SimpleNN, EarlyStopping, log_metrics
from sklearn.metrics import roc_auc_score, f1_score, precision_score

def signal_handler(sig, frame):
    print('Terminating the server...', flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def aggregate(global_model, client_state_dicts):
    global_state_dict = global_model.state_dict()
    for k in global_state_dict.keys():
        global_state_dict[k] = torch.mean(torch.stack([client[k] for client in client_state_dicts]), dim=0)
    global_model.load_state_dict(global_state_dict)
    return global_model

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, y.argmax(dim=1)).item()
        predictions = outputs.argmax(dim=1).numpy()
        targets = y.argmax(dim=1).numpy()
        accuracy = (predictions == targets).mean()
        roc_auc = roc_auc_score(targets, outputs[:, 1].numpy())
        f1 = f1_score(targets, predictions)
        precision = precision_score(targets, predictions)
    return loss, accuracy, roc_auc, f1, precision

def concatenate_metrics(df, metrics_dict):
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df = metrics_df.dropna(axis=1, how='all')
    df = pd.concat([df, metrics_df], ignore_index=True)
    return df

def main(experiment_name, num_clients, num_rounds, batch_size):
    print("-----------------------------------------", flush=True)
    print('Server initialization has started....!!!', flush=True)
    print("-----------------------------------------", flush=True)

    data_path = "./data/centralized_test_data.csv"
    X, y, fixed_input_size = preprocess_dataset(data_path)

    global_model = SimpleNN(fixed_input_size)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=FLconfig.LEARNING_RATE)
    early_stopping = EarlyStopping(patience=FLconfig.patience)  

    experiment_dir = f"./experiments/{experiment_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    weights_dir = os.path.join(experiment_dir, "server_weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    input_size_path = os.path.join(experiment_dir, "input_size.txt")
    with open(input_size_path, "w") as f:
        f.write(str(fixed_input_size))

    metrics_file = os.path.join(experiment_dir, "communication_metrics.csv")
    logs_file = os.path.join(experiment_dir, "logs.csv")

    for server_round in range(num_rounds):
        print(f"Starting round {server_round + 1}...", flush=True)

        num_sampled_clients = max(5, int(FLconfig.fraction_of_clients * num_clients))
        selected_clients = random.sample(range(num_clients), num_sampled_clients)

        client_updates = []
        timeout_count = 0
        total_data_transferred = 0
        total_comm_time = 0
        total_client_compute_time = 0
        round_start_time = time.time()

        for client_id in selected_clients:
            client_start_time = time.time()
            try:
                client_state_dict_path = os.path.join(experiment_dir, f"client_{client_id}", "weights", f"round_{server_round}.pth")
                max_wait_time = 20
                wait_time = 0
                while not os.path.exists(client_state_dict_path) and wait_time < max_wait_time:
                    time.sleep(1)
                    wait_time += 1
                
                if os.path.exists(client_state_dict_path):
                    client_data = torch.load(client_state_dict_path)
                    state_dict = client_data['state_dict']
                    client_compute_time = client_data['compute_time']
                    client_updates.append((state_dict, client_id))
                    total_data_transferred += os.path.getsize(client_state_dict_path) / (1024 * 1024)  # Convert bytes to MB
                    total_client_compute_time += client_compute_time
                else:
                    timeout_count += 1
            except Exception as e:
                print(f"Error loading client {client_id} update: {e}", flush=True)
                timeout_count += 1
            client_end_time = time.time()
            total_comm_time += (client_end_time - client_start_time)

        if timeout_count > 0:
            print(f"Timeout occurred for {timeout_count} clients in round {server_round + 1}", flush=True)

        if not client_updates:
            print(f"No client updates received for round {server_round + 1}. Continuing with previous model.", flush=True)
            continue

        aggregate_start_time = time.time()
        global_model = aggregate(global_model, [state_dict for state_dict, _ in client_updates])
        server_aggregate_time = time.time() - aggregate_start_time

        if (server_round + 1) % FLconfig.checkpoint_interval == 0:
            checkpoint_path = os.path.join(weights_dir, f"round_{server_round + 1}.pth")
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at round {server_round + 1}: {checkpoint_path}", flush=True)

        loss, accuracy, roc_auc, f1, precision = evaluate_model(global_model, X, y)
        print(f"Round {server_round + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}", flush=True)

        round_end_time = time.time()
        round_total_time = round_end_time - round_start_time
        avg_rtt = total_comm_time / num_sampled_clients if num_sampled_clients else 0

        log_metrics(metrics_file, server_round + 1, avg_rtt, total_data_transferred, num_sampled_clients, total_comm_time, total_client_compute_time / num_sampled_clients if num_sampled_clients else 0, server_aggregate_time, round_total_time, accuracy)

        # Log performance metrics
        if os.path.isfile(logs_file):
            df = pd.read_csv(logs_file)
        else:
            df = pd.DataFrame(columns=["round", "test_loss", "test_accuracy", "roc_auc", "f1_score", "precision"])

        metrics_dict = {
            "round": [server_round + 1],
            "test_loss": [loss],
            "test_accuracy": [accuracy],
            "roc_auc": [roc_auc],
            "f1_score": [f1],
            "precision": [precision]
        }
        df = concatenate_metrics(df, metrics_dict)
        df.to_csv(logs_file, index=False)
        print(f"Updated logs at: {logs_file}", flush=True)

        early_stopping(loss)
        if early_stopping.early_stop:
            print(f"Early stopping at round {server_round + 1} due to no improvement", flush=True)
            break

    print("Server aggregation completed", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", required=True, help="Experiment name")
    parser.add_argument("-n", "--num_clients", type=int, required=True, help="Number of clients")
    parser.add_argument("-r", "--num_rounds", type=int, required=True, help="Number of rounds")
    parser.add_argument("-b", "--batch_size", type=int, required=True, help="Batch size")
    args = parser.parse_args()

    main(args.experiment_name, args.num_clients, args.num_rounds, args.batch_size)
