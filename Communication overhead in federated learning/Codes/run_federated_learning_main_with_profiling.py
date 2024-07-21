#run_federated_learning_main_with_profiling.py

import os
import time
import argparse

def main(num_clients, batch_size):
    # Generate a unique identifier for the experiment
    experiment_id = f"{time.strftime('%Y%m%d_%H%M%S')}_clients{num_clients}_batch{batch_size}"

    # Define the profiling directory
    profiling_dir = os.path.join("/global/homes/b/billmj/fl_unsw", f"scorep_{experiment_id}")

    # Create the directory
    os.makedirs(profiling_dir, exist_ok=True)

    # Run the federated learning script with Score-P
    os.system(f"SCOREP_PROFILING_FORMAT=cube3 SCOREP_EXPERIMENT_DIRECTORY={profiling_dir} scorep python federated_learning_main.py {num_clients} {batch_size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning experiment with profiling")
    parser.add_argument("num_clients", type=int, help="Number of clients")
    parser.add_argument("batch_size", type=int, help="Batch size")
    args = parser.parse_args()

    main(args.num_clients, args.batch_size)
