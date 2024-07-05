#!/bin/bash

# Function to handle cleanup on termination
cleanup() {
  echo "Terminating all subprocesses..."
  kill -- -$$
  exit 1
}

# Trap SIGINT and call cleanup
trap cleanup SIGINT

if [ $# -ne 3 ]; then
  echo "Usage: ./run.sh <number_of_clients> <num_rounds> <batch_size>"
  exit 1
fi

num_clients=$1
num_rounds=$2
batch_size=$3

dir="./data/client_dataset"

if [ ! -d "$dir" ]; then
  echo "Directory $dir does not exist."
  exit 1
fi

if [ ! -f "server.py" ]; then
  echo "server.py not found!"
  exit 1
fi

if [ ! -f "client.py" ]; then
  echo "client.py not found!"
  exit 1
fi

files=()

for file in $dir/*; do
  if [[ -f $file && $file == *.csv ]]; then
    files+=("$file")
  fi
done

if [ ${#files[@]} -eq 0 ]; then
  echo "No CSV files found in the directory."
  exit 1
fi

echo "Files to be processed: ${files[@]}"

experiment_name="baseline_experiment_${num_clients}clients_${num_rounds}rounds_${batch_size}batchsize"

# Start the server
python server.py -e $experiment_name -n $num_clients -r $num_rounds -b $batch_size > server_output.log 2>&1 &
server_pid=$!
echo "Server started with PID $server_pid"

sleep 5

# Display initial server output
cat server_output.log

# Wait for input_size.txt to be created
while [ ! -f "./experiments/${experiment_name}/input_size.txt" ]; do
  sleep 1
done

fixed_input_size=$(cat "./experiments/${experiment_name}/input_size.txt")

# Start the clients
for i in $(seq 0 $((num_clients - 1))); do
    echo "Starting client $i with file ${files[i]}"
    python client.py -p "${files[i]}" -c "$i" -e $experiment_name -r $num_rounds --epochs 5 --lr 0.001 --batch_size $batch_size --fixed_input_size $fixed_input_size &
    client_pids[$i]=$!
done

# Wait for all client processes
for pid in ${client_pids[@]}; do
    wait $pid
done

# Wait for the server process
wait $server_pid
echo "Server process completed. Server output:"
cat server_output.log

# Check if the logs and checkpoints are created
echo "Checking for logs and checkpoints..."
ls -l ./experiments/${experiment_name}/
ls -l ./experiments/${experiment_name}/server_weights/
cat ./experiments/${experiment_name}/logs.csv

# Cleanup on exit
cleanup
