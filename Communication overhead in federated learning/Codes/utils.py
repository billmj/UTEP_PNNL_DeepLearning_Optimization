#utils.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    matthews_corrcoef
)

def load_data():
    # Load and preprocess data
    data = pd.read_csv('/global/homes/b/billmj/fl_unsw/data/centralized_test_data.csv')
    
    # Separate features and target
    X = data.drop(columns=['label'])
    y = data['label']
    
    # Identify categorical columns
    categorical_columns = ['proto', 'service', 'state']
    
    # Apply one-hot encoding to categorical columns and scaling to numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])
    
    X = preprocessor.fit_transform(X)
    
    # Convert sparse matrix to dense format
    X = X.toarray()
    
    # Print the type and shape of the transformed data
    print("Type of X after transformation:", type(X))
    print("Shape of X after transformation:", X.shape)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to numpy arrays (should be already numpy arrays, but this is to ensure)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Print shapes to verify
    print("Final X_train shape:", X_train.shape)
    print("Final X_test shape:", X_test.shape)
    print("Final y_train shape:", y_train.shape)
    print("Final y_test shape:", y_test.shape)
    
    return X_train, y_train, X_test, y_test

# Experiment directory creation
def create_experiment_dir(exp_name):
    base_dir = "experiments"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

# Metrics saving
def save_metrics(metrics, filename, exp_dir):
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(exp_dir, filename), index=False)
    print(f"Metrics saved to {filename}")

# Model definition
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

# Helper functions for federated learning
def client_update(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    model.train()
    computation_time = 0
    for epoch in range(num_epochs):
        for data, target in train_loader:
            comp_start = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            comp_end = time.time()
            computation_time += comp_end - comp_start
        scheduler.step()
    return model.state_dict(), computation_time

def calculate_model_size(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # Size in MB

# Main federated learning function
def federated_learning_with_fedavg(X_train, y_train, X_test, y_test, num_clients, batch_size, num_rounds, num_epochs_per_round, initial_lr, checkpoint_interval, exp_dir):
    input_dim = X_train.shape[1]
    global_model = SimpleNN(input_dim)
    global_weights = global_model.state_dict()

    # Split data into clients
    client_data = []
    for i in range(num_clients):
        start = i * len(X_train) // num_clients
        end = (i + 1) * len(X_train) // num_clients
        client_data.append((X_train[start:end], y_train[start:end]))

    # Initialize the metrics
    communication_metrics = []
    performance_metrics = []

    for round in range(num_rounds):
        client_updates = []
        total_comm_time = 0
        total_comp_time = 0
        total_uplink_data = 0
        total_downlink_data = 0
        round_start_time = time.time()

        for client_x, client_y in client_data:
            client_tensor_x = torch.tensor(client_x, dtype=torch.float32)
            client_tensor_y = torch.tensor(client_y, dtype=torch.int64)
            client_dataset = TensorDataset(client_tensor_x, client_tensor_y)
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

            client_model = SimpleNN(input_dim)
            
            # Downlink: Global model to client
            downlink_start = time.time()
            client_model.load_state_dict(global_weights)
            downlink_end = time.time()
            downlink_time = downlink_end - downlink_start
            downlink_data = calculate_model_size(client_model)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(client_model.parameters(), lr=initial_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

            # Client computation
            new_weights, client_comp_time = client_update(client_model, client_loader, criterion, optimizer, scheduler, num_epochs_per_round)

            # Uplink: Client model to server
            uplink_start = time.time()
            client_updates.append(new_weights)
            uplink_end = time.time()
            uplink_time = uplink_end - uplink_start
            uplink_data = calculate_model_size(client_model)

            total_comm_time += downlink_time + uplink_time
            total_comp_time += client_comp_time
            total_uplink_data += uplink_data
            total_downlink_data += downlink_data

        # Average updates from all clients (FedAvg)
        global_weights = {k: torch.stack([client[k] for client in client_updates], 0).mean(0) for k in global_weights.keys()}
        global_model.load_state_dict(global_weights)

        # Save checkpoint
        if (round + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(exp_dir, f'round_{round+1}.pth')
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at round {round + 1}")

        # Evaluate global model
        round_metrics = evaluate_model(global_model, X_test, y_test)
        round_metrics = {'Round': round + 1, **round_metrics}
        performance_metrics.append(round_metrics)

        round_end_time = time.time()
        round_total_time = round_end_time - round_start_time
        avg_rtt = total_comm_time / num_clients

        # Log communication metrics
        comm_metrics = {
            "Round": round + 1,
            "Average_RTT(s)": avg_rtt,
            "Uplink_Data(MB)": total_uplink_data,
            "Downlink_Data(MB)": total_downlink_data,
            "Total_Data_Transferred(MB)": total_uplink_data + total_downlink_data,
            "Computation_Time(s)": total_comp_time,
            "Communication_Time(s)": total_comm_time,
            "Round_Total_Time(s)": round_total_time,
            "Round_Accuracy": round_metrics['Accuracy']
        }
        communication_metrics.append(comm_metrics)

        print(f"Round {round + 1}, Test accuracy: {round_metrics['Accuracy']:.4f}")

    # Save communication metrics
    save_metrics(communication_metrics, "communication_metrics.csv", exp_dir)
    
    # Save final performance metrics (including all rounds)
    save_metrics(performance_metrics, "final_metrics.csv", exp_dir)

    return global_model

# Model evaluation
def evaluate_model(model, X_test, y_test):
    model.eval()
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
    test_tensor_y = torch.tensor(y_test, dtype=torch.int64)
    with torch.no_grad():
        test_output = model(test_tensor_x).squeeze()
        y_pred = (test_output > 0.5).int()

    metrics = {
        "Accuracy": accuracy_score(test_tensor_y, y_pred),
        "F1_Score": f1_score(test_tensor_y, y_pred),
        "Precision": precision_score(test_tensor_y, y_pred),
        "Recall": recall_score(test_tensor_y, y_pred),
        "ROC_AUC": roc_auc_score(test_tensor_y, y_pred),
        "MCC": matthews_corrcoef(test_tensor_y, y_pred)
    }

    return metrics

    if round is not None:
        filename = f"metrics_round_{round}.csv"
    else:
        filename = "final_metrics.csv"

    save_metrics(metrics, filename, exp_dir)

    if round is None:
        print(f"Final Test accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(test_tensor_y, y_pred))

        print("\nClassification Report:")
        print(classification_report(test_tensor_y, y_pred))

    return accuracy
