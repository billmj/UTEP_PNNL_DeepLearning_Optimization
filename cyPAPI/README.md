The cyPAPI repository is here: https://github.com/icl-utk-edu/cyPAPI 
## Prerequisites

Ensure you have the following modules installed in your login node:
- module load papi
- module load python

Make sure you are on a compute node and place the account to the project name:
```bash
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=mXXXX
```
## Installation

Clone the repo into your home folder (optionally download and place zip folder to home directory using unzip bash command)

```bash
git clone https://github.com/icl-utk-edu/cyPAPI.git 
```
## Environment Setup

Before installing `cyPAPI`, set the required environment variables:
```bash
export PAPI_PATH=$CRAY_PAPI_PREFIX
export PAPI_CUDA_ROOT=$CUDA_HOME
```
Optionally - you can also place:

```bash
export PAPI_CUDA_ROOT=$CRAY_PAPI_PREFIX 
```
in case of potential errors.


Confirm the environment variables are set correctly:

```bash
ls $PAPI_CUDA_ROOT
ls $PAPI_PATH
```

## Building cyPAPI

1. **Pause DCGMI profiling**:
    ```bash
    dcgmi profile --pause
    ```

2. **Modify `cypapi.pyx` if necessary**:
    Ensure the following dictionary is defined before line 138:
    ```python
    PAPI_Error = {
        'PAPI_OK': PAPI_OK,
        'PAPI_EINVAL': PAPI_EINVAL,
        'PAPI_ENOMEM': PAPI_ENOMEM,
        'PAPI_ENOEVNT': PAPI_ENOEVNT,
        'PAPI_ENOCMP': PAPI_ENOCMP,
        'PAPI_EISRUN': PAPI_EISRUN,
        'PAPI_EDELAYINIT': PAPI_EDELAY_INIT
    }
    ```

3. **Build `cyPAPI`**:
    ```bash
    cd cyPAPI
    make clean
    make install
    ```

## Testing

To test if everything is working:

1. Navigate to the test directory:
    ```bash
    cd cyPAPI/cypapi/cytest
    ```

2. Modify `torch_cuda.py`:
    - Set `unit = "cuda_7011"`
    - Change instances using `unit` to `"cuda"`:
    ```python
    matrix_A = torch.rand(m_rows, n_cols, device="cuda")
    matrix_B = torch.rand(m_rows, n_cols, device="cuda")
    ```

3. Run the test file:
    ```bash
    ./torch_cuda.py
    ```

## Troubleshooting

Refer to the [NERSC documentation](https://docs.nersc.gov/environment/#:~:text=Many%20users%20include%20module%20load%20statements%20in%20their%20~/.bashrc%20to%20customize%20their%20startup%20modules%2C%20but%20this%20can%20cause%20unexpected%20side%2Deffects%20when%20loading%20other%20modules) for potential issues with environment variables.

If encountering an error such as Exception: PAPI Error -14: PAPI_start failed , make sure the dcgmi profile is paused. Optionally you can use the command:
```python 
python3 -m pdb torch_cuda.py
```
To debug the file line by line. Refer to the documentation how to use pdb https://docs.python.org/3/library/pdb.html 

## License

cyPAPI is licensed under the [BSD 2-Clause License](LICENSE).





