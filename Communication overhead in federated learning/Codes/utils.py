import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
import torch.optim as optim
import csv
import os

@dataclass
class FLconfig:
    LEARNING_RATE: float = 0.001  
    local_epochs: int = 5
    num_rounds: int = 20
    batch_size: int = 32
    specified_dir: str = "./experiments"
    checkpoint_interval: int = 5
    num_clients: int = 50
    fraction_of_clients: float = 0.6
    patience: int = 5
    top_n_clients: int = 6 
    categorical_columns_for_dataset = [
        "proto", "service", "state", "sttl", "dttl", "swin", "dwin", 
        "trans_depth", "ct_ftp_cmd", "is_ftp_login", "ct_srv_src", 
        "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", 
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_state_ttl", 
        "ct_flw_http_mthd"
    ]

def log_metrics(metrics_file, round_num, avg_rtt, total_data_transferred, comm_frequency, total_comm_time, client_compute_time, server_aggregate_time, round_total_time, model_accuracy):
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Average_RTT(s)", "Total_Data_Transferred(MB)", "Comm_Frequency", "Total_Communication_Time(s)", "Client_Compute_Time(s)", "Server_Aggregate_Time(s)", "Round_Total_Time(s)", "Model_Accuracy"])
    with open(metrics_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round_num, avg_rtt, total_data_transferred, comm_frequency, total_comm_time, client_compute_time, server_aggregate_time, round_total_time, model_accuracy])

def preprocess_dataset(path_to_csv, fixed_input_size=None):
    df = pd.read_csv(path_to_csv)
    
    X = df.drop(columns=["label"])
    y = pd.get_dummies(df["label"]).values
    
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(X[numerical_columns])
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_categorical = encoder.fit_transform(X[categorical_columns])
    
    X_processed = np.hstack((X_numerical, X_categorical))
    
    if fixed_input_size is None:
        fixed_input_size = X_processed.shape[1]
    
    if X_processed.shape[1] < fixed_input_size:
        padding = np.zeros((X_processed.shape[0], fixed_input_size - X_processed.shape[1]))
        X_processed = np.hstack((X_processed, padding))
    elif X_processed.shape[1] > fixed_input_size:
        X_processed = X_processed[:, :fixed_input_size]
    
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor, fixed_input_size

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(64, 32)
        self.dropout3 = torch.nn.Dropout(0.3)
        self.fc4 = torch.nn.Linear(32, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.softmax(self.fc4(x), dim=1)
        return x
