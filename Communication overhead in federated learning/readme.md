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
  
### Installing Dependencies

2. **Using Conda**:
   If you are using Conda, you can create a new environment and install the dependencies from the `requirements.txt` file by running the following commands:

   ```shell
   conda create --name your_env_name python=3.9
   conda activate your_env_name
   pip install -r requirements.txt

### 3. Baseline Run

**Execute Existing Code:**
- Run federated learning code on the UNSW dataset.

**Collect Baseline Data:**
- Measure and record communication overhead metrics such as latency, bandwidth usage, and total communication time.
- Use logging and profiling tools for data collection.

### 4. Profiling and Analysis

**Profiling Tools:**
- Use HPC profiling tools to analyze communication patterns:
  - **Score-P**: Install, configure, and instrument code to collect performance data.
  - **Tau**: Install, configure, and instrument code for detailed performance analysis.

**Identify Bottlenecks:**
- Analyze profiling data to identify major sources of communication overhead.
- Focus on functions or communication calls that take the most time or resources.

### 5. Optimization Strategies

**Data Parallelism:**
- Distribute the dataset across multiple nodes.
- Implement using frameworks like TensorFlow or PyTorch’s distributed package.

**Model Parallelism:**
- Divide the neural network model across different nodes.
- Use model parallelism techniques available in deep learning frameworks.

### 6. Performance Modeling

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

## Communication Metrics in baseline

**Average_RTT (s):**
   Average Round Trip Time in seconds, representing the average time for a complete client-server communication cycle.

**Uplink_Data(MB):**
   The amount of data in megabytes sent from clients to the server in this round.

**Downlink_Data(MB):**
   The amount of data in megabytes sent from the server to clients in this round.

**Total_Data_Transferred(MB):**
  The total amount of data transferred (both uplink and downlink) in megabytes for this round.

**Computation_Time(s):**
   The time spent on local computations by clients in seconds for this round.

**Communication_Time(s):**
   The time spent on communication between clients and server in seconds for this round.

**Round_Total_Time(s):**
   The total time taken to complete this round, including both computation and communication.
     
## Getting Started (baseline experiments)


### Step 1: To run baseline experiments, specify the number of clients, number of rounds, and batch size as arguments:
```sh
python federated_learning_main.py <num_clients> <batch_size> #Replace `<num_clients>` with the number of clients (e.g., 10, 50, 100) and `<batch_size>` with the desired batch size (e.g., 32, 64, 128, 256).

```

Pressing Ctrl+C will terminate all processes




## Next Steps/Plans

- Perform detailed profiling to identify bottlenecks in the current system.
- Start GPU optimizations to accelerate computations and further reduce communication overhead.

## Acknowledgments

This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award ASCR-ERCAP0030084.
