# utils_ddp.py

import os
import time
import csv
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
import torch.distributed as dist
import matplotlib.pyplot as plt

# Load and preprocess the dataset
def load_data():
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

# Directory creation for experiments
def create_experiment_dir(exp_name):
    base_dir = "experiments_ddp"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

# Model definition using a simple neural network
class SimpleNN(nn.Module):
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
        x = torch.sigmoid(self.fc3(x))
        return x

def adjust_learning_rate(num_clients, initial_lr):
    return initial_lr * np.sqrt(num_clients)

# Model training function
def client_update(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.unsqueeze(1).float())  # Make sure the loss function is used here
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        epoch_losses.append(running_loss / len(train_loader))  # Average loss for the epoch

    return model.state_dict(), epoch_losses

# Initialize DDP process group
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def save_metrics(metrics, filename, exp_dir):
    """
    Save performance metrics to a CSV file.
    Args:
    metrics (list of dicts): The performance metrics to save.
    filename (str): The name of the file to save the metrics to.
    exp_dir (str): The directory to save the file in.
    """
    file_path = os.path.join(exp_dir, filename)
    headers = metrics[0].keys()

    with open(file_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Metrics saved to {file_path}")

def plot_learning_curve(performance_metrics, exp_dir, round_losses):
    rounds = [metric['Round'] for metric in performance_metrics]
    accuracies = [metric['Accuracy'] for metric in performance_metrics]

    # Create a figure with dual y-axis (one for accuracy, one for loss)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot accuracy on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Test Accuracy', color=color)
    ax1.plot(rounds, accuracies, marker='o', color=color, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([min(accuracies) - 0.01, max(accuracies) + 0.01])

    # Create a second y-axis for the loss values
    ax2 = ax1.twinx()  # Instantiate a second axis sharing the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Average Training Loss', color=color)
    ax2.plot(rounds, round_losses, marker='x', color=color, linestyle='--', label='Average Training Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title and grid
    plt.title('Learning Curve: Test Accuracy and Training Loss vs. Rounds')
    fig.tight_layout()

    # Create a subfolder for visualizations
    vis_dir = os.path.join(exp_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Save the learning curve plot with both accuracy and loss
    plt.savefig(os.path.join(vis_dir, 'learning_curve_with_loss.png'))
    plt.close()

# Main federated learning function with Distributed Data Parallel (DDP)
def federated_learning_with_ddp(X_train, y_train, X_test, y_test, num_clients, batch_size, num_rounds, num_epochs_per_round, initial_lr, checkpoint_interval, exp_dir, criterion, rank, world_size):
    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # Split data into clients
    input_dim = X_train.shape[1]
    global_model = SimpleNN(input_dim).to(device)

    # Wrap the model in DistributedDataParallel
    ddp_model = DDP(global_model, device_ids=[rank])

    # Prepare clients' data
    client_data = []
    for i in range(num_clients):
        start = i * len(X_train) // num_clients
        end = (i + 1) * len(X_train) // num_clients
        client_data.append((X_train[start:end], y_train[start:end]))

    # Initialize metrics
    communication_metrics = []
    performance_metrics = []
    round_losses = []  # To store average loss per round

    for round in range(num_rounds):
        # Adjust the learning rate based on the number of clients
        current_lr = adjust_learning_rate(num_clients, initial_lr)
        
        # Define optimizer with adjusted learning rate
        optimizer = optim.Adam(ddp_model.parameters(), lr=current_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        client_updates = []
        client_round_losses = []  # To store losses for each client in the round
        round_start_time = time.time()

        for client_x, client_y in client_data:
            client_tensor_x = torch.tensor(client_x, dtype=torch.float32)
            client_tensor_y = torch.tensor(client_y, dtype=torch.int64)
            client_dataset = TensorDataset(client_tensor_x, client_tensor_y)

            # Distributed data sampler for DDP
            sampler = DistributedSampler(client_dataset, num_replicas=world_size, rank=rank)
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

            # Update model on each client
            new_weights, epoch_losses = client_update(ddp_model, client_loader, criterion, optimizer, scheduler, num_epochs_per_round, device)
            client_updates.append(new_weights)
            client_round_losses.append(np.mean(epoch_losses))  # Track average loss for each client

        # FedAvg - Average updates from clients
        global_weights = {k: torch.stack([client[k].to(device) for client in client_updates], 0).mean(0) for k in client_updates[0].keys()}
        ddp_model.load_state_dict(global_weights)

        # Average losses across all clients for the round
        avg_round_loss = np.mean(client_round_losses) if client_round_losses else 0.0
        round_losses.append(avg_round_loss)  # Append the average loss for the round

        # Save checkpoints
        if (round + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(exp_dir, f'round_{round+1}.pth')
            torch.save(ddp_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at round {round + 1}")

        # Evaluate global model
        round_metrics = evaluate_model(ddp_model, X_test, y_test, device)
        round_metrics['Round'] = round + 1
        performance_metrics.append(round_metrics)

        round_end_time = time.time()
        
        # Print detailed log for this round
        print(f"Round {round + 1}")
        print(f"Current learning rate: {current_lr:.6f}")
        print(f"Batch size: {batch_size}")
        print(f"Number of clients: {num_clients}")
        print(f"Average loss: {avg_round_loss:.4f}")
        print(f"Test accuracy: {round_metrics['Accuracy']:.4f}")
        print(f"Time taken: {round_end_time - round_start_time:.2f}s")
        print("-----------------------------------")

    # Save final metrics
    save_metrics(performance_metrics, "final_metrics_ddp.csv", exp_dir)

    cleanup_ddp()
    
    return ddp_model, performance_metrics, round_losses  # Return round_losses as well




def evaluate_model(model, X_test, y_test, device):
    model.eval()  # Set the model to evaluation mode
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_tensor_y = torch.tensor(y_test, dtype=torch.int64).to(device)
    with torch.no_grad():  # No need for gradients during evaluation
        test_output = model(test_tensor_x).squeeze()
        y_pred = (test_output > 0.5).int()

    # Move tensors back to CPU for metric calculation
    y_pred = y_pred.cpu()
    test_tensor_y = test_tensor_y.cpu()

    # Calculate and return performance metrics
    metrics = {
        "Accuracy": accuracy_score(test_tensor_y, y_pred),
        "F1_Score": f1_score(test_tensor_y, y_pred),
        "Precision": precision_score(test_tensor_y, y_pred),
        "Recall": recall_score(test_tensor_y, y_pred),
        "ROC_AUC": roc_auc_score(test_tensor_y, y_pred),
        "MCC": matthews_corrcoef(test_tensor_y, y_pred)
    }

    return metrics
