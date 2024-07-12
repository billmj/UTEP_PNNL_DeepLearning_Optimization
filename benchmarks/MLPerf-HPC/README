## Prerequisites
Clone the repo at https://github.com/mlcommons/hpc_results_v3.0/ 
cd into hpc_results_v3.0/HPE+LBNL/benchmarks 

You do not need to load any modules because each benchmark runs in a shifter container. You should not have to change the shifter image provided in the config files.

## Getting the Data
The data is currently stored in our Community File System which can be accessed by: 
```bash
cd $CFS/<your_project_name> 
```
Alternatively you can download the data directly from the following globus endpoints:
- cosmoflow -- https://app.globus.org/file-manager?origin_id=4747b7f1-3bc5-40d2-85a4-c498faccb2c3&origin_path=%2F 
- deepcam -- https://app.globus.org/file-manager?origin_id=5cd63a0c-b684-4a6e-8d3b-4c0ce690661c&origin_path=%2F
- opencatalyst -- https://app.globus.org/file-manager?origin_id=7008326a-def1-11eb-9b4d-47c0f9282fb8&origin_path=%2F

It is advised that we run the data from our personal scratch directories. To move the data from community to scratch, it is easier to use Globus. 

### How to use Globus
Create a free account at https://www.globus.org/ using your nersc information. This will link nersc to globus as one of the many identities you can use with globus. You can link other sites to globus including github through the settings option.

After you have successfully created an account with globus you can transfer the data using the file manager tab. You can either follow the links above directly or enter the collection and path for the community directory. If you are transferring data within perlmutter the collection is: NERSC Perlmutter and the path to where the data is stored and being transferred to.

For example:

![Globus example](https://photos.app.goo.gl/xafoTKvQ3SmzJmQ36)

You then need to select the start button on the panel in which you are transferring from. In the example above since I am transferring data from the community file system (panel on the left) to the scratch file system (panel on the right) I would press the start button on the left panel -- the community file system side.

If you do not see two pannels as shown above simply change the panel view in the top right hand corner of the page.

You do not need to stay logged into globus or Nersc while waiting for a data transfer.

For further information regarding Globus and Nersc usage please refer to here:

https://docs.nersc.gov/services/globus/

## Running the benchmarks
### Cosmoflow
To run cosmoflow on 128 GPU nodes using 4 A100 GPUs/node:
- cd cosmoflow/implementations/cosmoflow-pytorch
- use any preferred editor to change the account number in run_pm.sub
- then cd ./configs and change the data path in config_pm_common.sh

Following the instructions in perlmutter_128x4/README.md, return to `/cosmoflow-pytorch directory` and run with:

type:
```bash
source configs/config_pm_128x4x1.sh
sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
```

This will run cosmoflow using the train and validation data. If you wish to run it with a different configuration simply change the `config_pm_128x4x1.sh` to the desired configuration.

Follow a similar procedure to run the other MLPerf HPC benchmarks.

## Extra Notes
My cosmoflow job took at twice as long as the time reported at https://mlcommons.org/benchmarks/training-hpc/ , and running from my $SCRATCH directory didn't make much difference.

Paper from NVIDIA on optimizing MLPerf training on H100 GPUs
Leading MLPerf Training 2.1 with Full Stack Optimizations for AI
https://developer.nvidia.com/blog/leading-mlperf-training-2-1-with-full-stack-optimizations-for-ai/
