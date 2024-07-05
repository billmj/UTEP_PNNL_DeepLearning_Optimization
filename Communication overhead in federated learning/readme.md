# Communication Overhead Analysis in Distributed Deep Federated Learning

## Overview

This project aims to analyze and optimize communication overhead in federated learning systems, specifically within the context of distributed deep learning using the UNSW-NB15 dataset for network intrusion detection. The focus is on implementing efficient client-server communication protocols and optimizing data transfer processes to improve overall system performance.

## Technical Plan

### 1. Initial Setup

**Access NERSC’s Perlmutter:**
- Ensure active account and necessary permissions.
- Familiarize with user guides and documentation.

**Environment Setup:**
- Load Necessary Modules:
  - Python: `module load python`
  - MPI Libraries: `module load mpi`
- Install Dependencies:
  - Use a virtual environment or conda.
  - Install necessary Python libraries using requirements.txt

### 2. Baseline Run

**Execute Existing Code:**
- Run federated learning code on the UNSW dataset.

**Collect Baseline Data:**
- Measure and record communication overhead metrics such as latency, bandwidth usage, and total communication time.
- Use logging and profiling tools for data collection.

### 3. Profiling and Analysis

**Profiling Tools:**
- Use HPC profiling tools to analyze communication patterns:
  - **Score-P**: Install, configure, and instrument code to collect performance data.
  - **Tau**: Install, configure, and instrument code for detailed performance analysis.

**Identify Bottlenecks:**
- Analyze profiling data to identify major sources of communication overhead.
- Focus on functions or communication calls that take the most time or resources.

### 4. Optimization Strategies

**Data Parallelism:**
- Distribute the dataset across multiple nodes.
- Implement using frameworks like TensorFlow or PyTorch’s distributed package.

**Model Parallelism:**
- Divide the neural network model across different nodes.
- Use model parallelism techniques available in deep learning frameworks.

### 5. Performance Modeling

**Simulate Different Strategies:**
- Use performance modeling tools to simulate various optimization strategies:
  - **Score-P and Cube**: Collect and visualize profiling data to understand the impact of different optimizations.

**Develop Models:**
- Develop performance models based on simulation results to predict the efficiency of different strategies.

## Current Progress

- **Baseline Experiments:** Conducted with varying client numbers and batch sizes to measure initial performance.
- **Initial Findings:**
  - **System Performance:**
    - Efficient handling of varying numbers of clients (10 to 100) with reasonable communication overhead.
    - Consistent and manageable communication and computation times after the initial setup.
  - **Model Performance:**
    - Inconsistent accuracy ranging from 45% to 98%.
    - Wide variation in AUC ROC scores (0.46 to 0.99).
  - **Scalability vs. Performance:**
    - Efficient scaling in system performance with increased client numbers, but no clear improvement in model accuracy or AUC ROC scores.
  - **Learning Dynamics:**
    - Lack of consistent improvement over rounds.
    - Higher variability in accuracy with 10-client experiments compared to 50 and 100 clients.
  - **Communication vs. Performance:**
    - Good communication efficiency with reasonable data transfer sizes and communication times.
    - Non-linear increase in communication overhead as client numbers grow, suggesting effective management of distributed computation.

## Next Steps/Plans

- Perform detailed profiling to identify bottlenecks in the current system.
- Start GPU optimizations to accelerate computations and further reduce communication overhead.

## Getting Started

### Step 1: Ensure Datasets are Prepared
```sh
python create_dataset_for_clients.py -n 50  # Number of clients in this case for 50 clients
```

### Step 2: To run the experiment, specify the number of clients, number of rounds, and batch size as arguments:
```sh
./run.sh 50 20 32
```

Pressing Ctrl+C will terminate all processes
