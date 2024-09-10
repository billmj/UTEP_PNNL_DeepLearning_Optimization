# federated_learning_ddp.py

import argparse
from utils_ddp import run_federated_learning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DDP federated learning experiment")
    parser.add_argument("num_clients", type=int, help="Number of clients")
    parser.add_argument("batch_size", type=int, help="Batch size")
    args = parser.parse_args()

    world_size = 4  # Adjust based on number of GPUs or nodes
    run_federated_learning(world_size, args.num_clients, args.batch_size)
