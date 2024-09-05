#!/bin/bash
#SBATCH -N 1                   # Number of nodes
#SBATCH -C gpu                 # Use GPU nodes
#SBATCH -G 4                   # Number of GPUs per node
#SBATCH -q regular             # QoS (Quality of Service)
#SBATCH -J FL_DDP_Multi        # Job name
#SBATCH --mail-user=wmarfo@miners.utep.edu  # Your email to get notifications
#SBATCH --mail-type=ALL        # Send email on start, end, and failure
#SBATCH -t 03:00:00            # Wall time (HH:MM:SS) - increased to 3 hours
#SBATCH -A m4647               # Account to charge

# Load necessary modules and activate conda environment
module load python
source ~/.bashrc               # This ensures conda is initialized
conda activate fed_unsw        # Activate your conda environment

# Change to the correct directory
cd /global/homes/b/billmj/fl_unsw

# Set up environment variables for OpenMP
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Define arrays for client numbers and batch sizes
clients=(10 50 100)
batch_sizes=(32 64 128 256 512 1024 2048)

# Function to format time
format_time() {
    local seconds=$1
    printf "%dh:%dm:%ds" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

# Start of experiments
total_start_time=$(date +%s)
echo "Starting multi-configuration DDP Federated Learning experiments at $(date)"

# Loop through different configurations
for num_clients in "${clients[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        echo "-----------------------------------"
        echo "Running experiment with $num_clients clients and batch size $batch_size"
        start_time=$(date +%s)
        
        if srun python federated_learning_ddp.py $num_clients $batch_size; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            formatted_time=$(format_time $duration)
            echo "Finished experiment with $num_clients clients and batch size $batch_size"
            echo "Time taken: $formatted_time"
        else
            echo "Error occurred with $num_clients clients and batch size $batch_size"
        fi
        
        echo "-----------------------------------"
    done
done

# End of experiments
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
formatted_total_time=$(format_time $total_duration)
echo "All experiments completed at $(date)"
echo "Total time taken: $formatted_total_time"