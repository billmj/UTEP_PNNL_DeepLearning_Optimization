#!/bin/bash
#SBATCH -N 1                   # Number of nodes
#SBATCH -C gpu                 # Use GPU nodes
#SBATCH -G 4                   # Number of GPUs per node
#SBATCH -q regular             # QoS (Quality of Service)
#SBATCH -J FL_DDP_NsightSys    # Job name
#SBATCH --mail-user=wmarfo@miners.utep.edu  # Your email to get notifications
#SBATCH --mail-type=ALL        # Send email on start, end, and failure
#SBATCH -t 01:00:00            # Wall time (HH:MM:SS) - 1 hour (adjust as needed)
#SBATCH -A m4647               # Account to charge

# Load necessary modules and activate conda environment
module load python cudatoolkit
source ~/.bashrc               # Ensure conda is initialized
conda activate fed_unsw        # Activate your conda environment

# Change to the correct directory
cd /global/homes/b/billmj/fl_unsw

# Set up environment variables for OpenMP
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Create a directory for Nsight Systems output
export NSIGHT_OUTPUT_DIRECTORY=fl_ddp_profile_nsight
mkdir -p $NSIGHT_OUTPUT_DIRECTORY

# Run Nsight Systems profiling
srun nsys profile --stats=true \
                  -t nvtx,cuda \
                  --cuda-memory-usage=true \
                  -o $NSIGHT_OUTPUT_DIRECTORY/fl_profile_%h_%p \
                  python federated_learning_ddp.py 10 32
