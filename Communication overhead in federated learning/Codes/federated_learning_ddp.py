# federated_learning_ddp.py

import argparse
import time
import torch
from utils_ddp import load_data, create_experiment_dir, federated_learning_with_ddp, evaluate_model, plot_learning_curve
import torch.multiprocessing as mp
import torch.nn as nn  # Add this import for defining criterion

def main(rank, world_size, num_clients, batch_size):
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_data()

    # Create experiment directory
    exp_name = f"ddp_experiment_{num_clients}clients_{batch_size}batchsize"
    exp_dir = create_experiment_dir(exp_name)

    # Define the loss function (binary cross-entropy loss for binary classification)
    criterion = nn.BCELoss()

    # Run federated learning
    start_time = time.time()
    
    # Capture performance metrics and round_losses for all rounds
    ddp_model, performance_metrics, round_losses = federated_learning_with_ddp(  # Note round_losses added
        X_train, y_train, X_test, y_test,
        num_clients=num_clients,
        batch_size=batch_size,
        num_rounds=15,
        num_epochs_per_round=5,
        initial_lr=0.001,
        checkpoint_interval=5,
        exp_dir=exp_dir,
        criterion=criterion,
        rank=rank,
        world_size=world_size
    )

    # Final evaluation using the trained model
    device = torch.device(f"cuda:{rank}")
    final_metrics = evaluate_model(ddp_model, X_test, y_test, device)
    
    print("\nFinal Evaluation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    # Plot learning curve (with both performance_metrics and round_losses)
    plot_learning_curve(performance_metrics, exp_dir, round_losses)  # Add round_losses here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DDP federated learning experiment")
    parser.add_argument("num_clients", type=int, help="Number of clients")
    parser.add_argument("batch_size", type=int, help="Batch size")
    args = parser.parse_args()

    world_size = 4  # Adjust based on number of GPUs or nodes
    mp.spawn(main, args=(world_size, args.num_clients, args.batch_size), nprocs=world_size, join=True)
