# Communication Overhead Analysis in Distributed Deep Federated Learning

## Overview

This project aims to analyze and optimize communication overhead in federated learning systems, specifically within the context of distributed deep learning using the UNSW-NB15 dataset for network intrusion detection. The focus is on implementing efficient client-server communication and optimizing data transfer processes to improve overall system performance.

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
- Measure and record communication overhead metrics such as average round trip time (RTT) in seconds, uplink and downlink data transfer in megabytes (MB), computation and communication time per round in seconds.
- Track model performance metrics such as accuracy per round as well as final accuracy with early stopping, F1-score, precision, recall, ROC AUC, and Matthews Correlation Coefficient (MCC).

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

## Current Progress in baseline model

All baseline models were run using the Exclusive CPU Node on Perlmutter, equipped with AMD EPYC 7763 64-Core Processor (256 CPUs) and 503 GiB of memory, to ensure consistent performance.

Conducted federated learning experiments on the UNSW-NB15 dataset for network intrusion detection, exploring various configurations:

- Number of clients: 10, 50, 100
- Batch sizes: 32, 64, 128, 256

Key findings:

1. Learning Curves:
   - Smaller batch sizes (32, 64) converge faster and achieve higher accuracies.
   - Fewer clients (10) generally yield better performance.
   - Best configuration: 10 clients, batch size 32 (Accuracy: 0.986)
   - Worst configuration: 100 clients, batch size 256 (Accuracy: 0.551)

2. ROC AUC Performance:
   - Best configuration: 10 clients, batch size 32 (ROC AUC: 0.9838)

3. Computation vs. Communication Time:
   - Computation time significantly outweighs communication time across all configurations.
   - Larger batch sizes reduce computation time but have minimal impact on communication time.

4. Efficiency (Accuracy Gain per MB):
   - Configurations with fewer clients (10) show higher efficiency.
   - Smaller batch sizes tend to be more efficient in terms of accuracy gain per data transferred.

5. Scalability:
   - Larger batch sizes (128, 256) scale better with increasing client numbers.
   - Smaller batch sizes (32, 64) show poorer scalability as client numbers increase.

The baseline results provide insights into the trade-offs between model performance, communication efficiency, and scalability in federated learning setups.

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
