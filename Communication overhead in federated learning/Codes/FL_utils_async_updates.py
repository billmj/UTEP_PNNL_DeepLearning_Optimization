import os
import time
import csv
import socket
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch.cuda.nvtx as nvtx
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_free_port():
    """Finds a free port to set up communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup_ddp(rank, world_size):
    """Sets up Distributed Data Parallel (DDP) environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('SLURM_STEP_NODELIST', 'localhost')
    os.environ['MASTER_PORT'] = os.environ['SLURM_STEP_RESV_PORTS'].split('-')[0]  # Use the first reserved port
    print(f"Rank {rank}: Using MASTER_ADDR={os.environ['MASTER_ADDR']} and MASTER_PORT={os.environ['MASTER_PORT']}")

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '0'
    os.environ['NCCL_NET_GDR_LEVEL'] = '5'

    try:
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30)
        )
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: Process group initialized successfully")
    except Exception as e:
        print(f"Rank {rank}: Error initializing process group: {str(e)}")
        raise

    dist.barrier()
    print(f"Rank {rank}: Passed barrier")

def load_data():
    """Loads and preprocesses the dataset."""
    data = pd.read_csv('/global/homes/b/billmj/fl_unsw/data/centralized_test_data.csv')
    X = data.drop(columns=['label'])
    y = data['label']
    categorical_columns = ['proto', 'service', 'state']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])
    X = preprocessor.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def create_experiment_dir(exp_name):
    """Creates a directory for experiment logs and checkpoints."""
    base_dir = "experiments_ddp"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

class SimpleNN(nn.Module):
    """A simple neural network model for anomaly detection."""
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)  # Removed sigmoid for stability during loss calculation

def adjust_learning_rate(num_clients, initial_lr):
    """Adjusts learning rate based on the number of clients."""
    return initial_lr * np.sqrt(num_clients)

def client_update(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Trains the model on client-side data using mixed precision."""
    
    scaler = GradScaler()
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Enable mixed precision with autocast
            with autocast():
                output = model(data)
                loss = criterion(output, target.unsqueeze(1).float())

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            
            # Update optimizer with scaled gradients
            scaler.step(optimizer)
            
            # Update the scaler
            scaler.update()
            
            running_loss += loss.item()

        scheduler.step()
        epoch_losses.append(running_loss / len(train_loader))

    return model.state_dict(), epoch_losses

def compare_updates(local_update, global_update, relevance_threshold=0.65):
    """Compares local and global updates using selectve gradient alignment."""
    aligned_params = 0
    total_params = 0

    for key in local_update.keys():
        # Compare the signs of the local and global updates
        local_grad = torch.sign(local_update[key])
        global_grad = torch.sign(global_update[key])

        aligned_params += (local_grad == global_grad).sum().item()
        total_params += local_grad.numel()

    # Calculate percentage of aligned parameters
    alignment_ratio = aligned_params / total_params

    # Return True if the alignment exceeds the threshold, False otherwise
    return alignment_ratio >= relevance_threshold

def async_client_update(client_idx, model, client_data, criterion, optimizer, scheduler, num_epochs, device, global_weights):
    """Handles asynchronous client updates with CMFL-based relevance filtering."""
    client_tensor_x = torch.tensor(client_data[0], dtype=torch.float32)
    client_tensor_y = torch.tensor(client_data[1], dtype=torch.int64)
    client_dataset = TensorDataset(client_tensor_x, client_tensor_y)

    client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)  # Batch size for clients
    
    new_weights, epoch_losses = client_update(model, client_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Compare local update with global weights using CMFL
    if compare_updates(new_weights, global_weights):
        print(f"Client {client_idx} update accepted (relevance threshold met).")
        return client_idx, new_weights, np.mean(epoch_losses)
    else:
        print(f"Client {client_idx} update rejected (relevance threshold not met).")
        return client_idx, None, np.mean(epoch_losses)

def federated_learning_with_async_comm_and_cmfl(rank, world_size, X_train, y_train, X_test, y_test, 
                                                num_clients, num_rounds, num_epochs, initial_lr, checkpoint_interval, exp_dir, criterion):
    device = torch.device(f"cuda:{rank}")
    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim).to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    client_data = []
    for i in range(num_clients):
        start = i * len(X_train) // num_clients
        end = (i + 1) * len(X_train) // num_clients
        client_data.append((X_train[start:end], y_train[start:end]))

    performance_metrics = []
    round_losses = []
    
    global_weights = model.state_dict()  # Initialize global weights

    for round in range(num_rounds):
        nvtx.range_push(f"Round {round + 1}")
        
        client_updates = []
        client_round_losses = []
        round_start_time = time.time()

        # Asynchronous client updates using a thread pool
        with ThreadPoolExecutor() as executor:
            futures = []
            for client_idx, data in enumerate(client_data):
                futures.append(executor.submit(async_client_update, client_idx, model, data, criterion, optimizer, scheduler, num_epochs, device, global_weights))

            for future in as_completed(futures):
                client_idx, new_weights, avg_loss = future.result()
                if new_weights is not None:  # Only aggregate relevant updates
                    client_updates.append(new_weights)
                client_round_losses.append(avg_loss)
                print(f"Client {client_idx} finished its update")

        # Aggregate the updates from clients that passed the CMFL relevance check
        if client_updates:
            global_weights = {k: torch.stack([client[k].to(device) for client in client_updates], 0).mean(0) for k in client_updates[0].keys()}
            model.load_state_dict(global_weights)

        avg_round_loss = np.mean(client_round_losses) if client_round_losses else 0.0
        round_losses.append(avg_round_loss)

        # Checkpoint saving
        if (round + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(exp_dir, f'round_{round+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at round {round + 1}")

        # Evaluate the model
        round_metrics = evaluate_model(model, X_test, y_test, device)
        round_metrics['Round'] = round + 1
        performance_metrics.append(round_metrics)

        round_end_time = time.time()
        print(f"Round {round + 1} | Avg loss: {avg_round_loss:.4f} | Test accuracy: {round_metrics['Accuracy']:.4f} | Time: {round_end_time - round_start_time:.2f}s")
        
        nvtx.range_pop()  # Round

    save_metrics(performance_metrics, "final_metrics_async_cmfl.csv", exp_dir)
    
    return model, performance_metrics, round_losses

def evaluate_model(model, X_test, y_test, device):
    """Evaluates the model on the test dataset."""
    model.eval()
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_tensor_y = torch.tensor(y_test, dtype=torch.int64).to(device)
    with torch.no_grad():
        test_output = model(test_tensor_x).squeeze()
        y_pred = (test_output > 0.5).int()

    y_pred = y_pred.cpu()
    test_tensor_y = test_tensor_y.cpu()

    metrics = {
        "Accuracy": accuracy_score(test_tensor_y, y_pred),
        "F1_Score": f1_score(test_tensor_y, y_pred),
        "Precision": precision_score(test_tensor_y, y_pred),
        "Recall": recall_score(test_tensor_y, y_pred),
        "ROC_AUC": roc_auc_score(test_tensor_y, y_pred),
        "MCC": matthews_corrcoef(test_tensor_y, y_pred)
    }

    return metrics

def cleanup_ddp():
    """Clean up the distributed process group."""
    try:
        dist.destroy_process_group()
        print("Process group destroyed")
    except Exception as e:
        print(f"Error during cleanup: {e}")
