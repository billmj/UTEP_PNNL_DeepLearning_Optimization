#!/bin/bash
#SBATCH -N 1                   # Number of nodes
#SBATCH -C gpu                 # Use GPU nodes
#SBATCH -G 4                   # Number of GPUs per node
#SBATCH -q regular             # QoS (Quality of Service)
#SBATCH -J FL_DDP_ScoreP       # Job name
#SBATCH --mail-user=wmarfo@miners.utep.edu  # Your email to get notifications
#SBATCH --mail-type=ALL        # Send email on start, end, and failure
#SBATCH -t 01:00:00            # Wall time (HH:MM:SS) - 1 hour (adjust as needed)
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

# Set up Score-P environment variables
export SCOREP_EXPERIMENT_DIRECTORY=fl_ddp_profile_scorep
export SCOREP_TOTAL_MEMORY=1G
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=200
export SCOREP_ENABLE_TRACING=true
export SCOREP_ENABLE_PROFILING=true
export SCOREP_METRIC_PAPI=PAPI_TOT_CYC,PAPI_TOT_INS
export SCOREP_METRIC_RUSAGE=ru_maxrss
export SCOREP_CUDA_ENABLE=yes
export SCOREP_CUDA_BUFFER=100M
export SCOREP_MPI_ENABLE=yes

# Run the profiled version
srun python -m scorep --nocompiler --thread=pthread federated_learning_ddp.py 10 32
