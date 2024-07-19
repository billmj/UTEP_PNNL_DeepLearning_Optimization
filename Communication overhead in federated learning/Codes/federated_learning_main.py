import argparse
import time
from utils import load_data, create_experiment_dir, federated_learning_with_fedavg, evaluate_model

def main(num_clients, batch_size):
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_data()

    # Create experiment directory
    exp_name = f"baseline_experiment_{num_clients}clients_{batch_size}batchsize"
    exp_dir = create_experiment_dir(exp_name)

    # Run federated learning
    start_time = time.time()
    final_model = federated_learning_with_fedavg(
        X_train, y_train, X_test, y_test,
        num_clients=num_clients,
        batch_size=batch_size,
        num_rounds=15,
        num_epochs_per_round=5,
        initial_lr=0.001,
        checkpoint_interval=5,
        exp_dir=exp_dir
    )

    # Final evaluation
    final_metrics = evaluate_model(final_model, X_test, y_test)
    print("\nFinal Evaluation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning experiment")
    parser.add_argument("num_clients", type=int, help="Number of clients")
    parser.add_argument("batch_size", type=int, help="Batch size")
    args = parser.parse_args()

    main(args.num_clients, args.batch_size)