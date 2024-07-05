# client.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import signal
import sys
from torch.utils.data import DataLoader, TensorDataset
from utils import FLconfig, preprocess_dataset, SimpleNN
import time

def signal_handler(sig, frame):
    print('Terminating the client...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.argmax(dim=1))
            loss.backward()
            optimizer.step()
    return model.state_dict()

def client_update(data, labels, model, experiment_dir, client_no, round_no, epochs, lr, batch_size):
    start_time = time.time()  # Start timing the computation
    dataset = TensorDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    new_state_dict = train(model, train_loader, criterion, optimizer, epochs)
    
    client_directory = os.path.join(experiment_dir, f"client_{client_no}")
    if not os.path.exists(client_directory):
        os.makedirs(client_directory)
    weights_directory = os.path.join(client_directory, "weights")
    if not os.path.exists(weights_directory):
        os.makedirs(weights_directory)
    
    compute_time = time.time() - start_time  # Compute time taken
    
    torch.save({
        'state_dict': new_state_dict,
        'compute_time': compute_time
    }, os.path.join(weights_directory, f"round_{round_no}.pth"))
    
    print(f"Client {client_no} update for round {round_no} completed")

def main(path_to_csv, client_no, experiment_name, num_rounds, epochs, lr, batch_size, fixed_input_size):
    data, labels, _ = preprocess_dataset(path_to_csv, fixed_input_size)
    
    model = SimpleNN(fixed_input_size)
    
    experiment_dir = f"./experiments/{experiment_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    for round_no in range(num_rounds):
        client_update(data, labels, model, experiment_dir, client_no, round_no, epochs, lr, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_to_csv", required=True, help="Path of CSV")
    parser.add_argument("-c", "--client_no", required=True, type=int, help="Client number")
    parser.add_argument("-e", "--experiment_name", required=True, help="Experiment name")
    parser.add_argument("-r", "--num_rounds", type=int, required=True, help="Number of rounds")
    parser.add_argument("--epochs", type=int, default=FLconfig.local_epochs, help="Number of epochs per round")
    parser.add_argument("--lr", type=float, default=FLconfig.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=FLconfig.batch_size, help="Batch size")
    parser.add_argument("--fixed_input_size", type=int, required=True, help="Fixed input size for the model")
    args = parser.parse_args()

    main(args.path_to_csv, args.client_no, args.experiment_name, args.num_rounds, args.epochs, args.lr, args.batch_size, args.fixed_input_size)
