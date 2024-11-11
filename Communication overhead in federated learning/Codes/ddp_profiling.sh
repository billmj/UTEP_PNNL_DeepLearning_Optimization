#!/bin/bash
#SBATCH -N 1                    # Number of nodes
#SBATCH -C gpu                  # Use GPU nodes
#SBATCH -G 4                    # Number of GPUs per node
#SBATCH --ntasks=4              # One task per GPU (4 tasks total)
#SBATCH --cpus-per-task=4       # Number of CPUs per task
#SBATCH -q regular              # QoS (Quality of Service)
#SBATCH -J FL_DDP_Multi         # Job name
#SBATCH --mail-user=wmarfo@miners.utep.edu  # Your email for notifications
#SBATCH --mail-type=ALL         # Send email on start, end, and failure
#SBATCH -t 02:00:00             # Total wall time (HH:MM:SS)
#SBATCH -A m4647                # Account to charge

# Load necessary modules and activate conda environment
module load python
source ~/.bashrc                # Ensure conda is initialized
conda activate fed_unsw          # Activate your conda environment

# Change to the correct directory
cd /global/homes/b/billmj/fl_unsw

# Set up environment variables for distributed computing
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export NCCL_DEBUG=INFO
export MASTER_ADDR=$(hostname)      # Use the hostname of the master node
export MASTER_PORT=$SLURM_STEP_RESV_PORTS  # Use the SLURM reserved port for this job
export WORLD_SIZE=4                 # Total number of tasks/ranks across all nodes

# Define arrays for different experiment configurations (overriding the Python defaults)
clients=(10)                         # Number of clients to test
batch_sizes=(512)                    # Batch sizes to test
epochs=(3)                           # Number of epochs per round
rounds=(10)                          # Number of rounds

# Function to format elapsed time
format_time() {
    local seconds=$1
    printf "%dh:%dm:%ds" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

# Start of experiments
total_start_time=$(date +%s)
echo "Starting multi-configuration DDP Federated Learning experiments at $(date)"

# Loop through client and batch size configurations
for num_clients in "${clients[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for epoch in "${epochs[@]}"
        do
            for round in "${rounds[@]}"
            do
                echo "-----------------------------------"
                echo "Running experiment with $num_clients clients, batch size $batch_size, $round rounds, and $epoch epochs per round"
                start_time=$(date +%s)
                
                # Use SLURM's srun to launch the Python script across all GPUs/ranks, passing parameters
                if srun --cpu_bind=cores python -u federated_learning_ddp.py $num_clients $batch_size $round $epoch; then
                    end_time=$(date +%s)
                    duration=$((end_time - start_time))
                    formatted_time=$(format_time $duration)
                    echo "Finished experiment with $num_clients clients, batch size $batch_size, $round rounds, and $epoch epochs"
                    echo "Time taken: $formatted_time"
                else
                    echo "Error occurred with $num_clients clients, batch size $batch_size, $round rounds, and $epoch epochs"
                fi
                
                echo "-----------------------------------"
            done
        done
    done
done

# End of experiments
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
formatted_total_time=$(format_time $total_duration)
echo "All experiments completed at $(date)"
echo "Total time taken: $formatted_total_time"