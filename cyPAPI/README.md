The cyPAPI repository is here: https://github.com/icl-utk-edu/cyPAPI 

## Note
The following steps are to install both cyPAPI and PAPI at your $HOME directory in perlmutter.

## Prerequisites

Ensure you have the following modules uninstalled/installed in your login node:
- module unload perftools-base
- module load python

Make sure you are on a compute node and place the account to the project name:
```bash
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=mXXXX
```
## Installation

Clone the papi and cypapi repo into your home folder (optionally download and place zip folder to home directory using unzip bash command)

```bash
git clone https://github.com/icl-utk-edu/cyPAPI.git 
git clone https://github.com/icl-utk-edu/papi.git 
```
## Environment Setup

Before installing `cyPAPI`, install `papi` and set the required environment variables (note you only need to set these varibles once for installing the software): 
```bash
export PAPI_PATH=/global/homes/<first initial of your username>/<perlmutter-username>
export PAPI_CUDA_ROOT=$CUDA_HOME
export CC=cc
export CXX=CC
export LIB="-lpthread -ldl"
export LIBS=$LIB
```
If you are using the papi module and perftools-base moudle - use:

```bash
export PAPI_PATH=$CRAY_PAPI_PREFIX
```

Then do the following commands under `papi/src`
```bash
./configure --prefix=$HOME --with-components="cuda"
make install
```

Before installing `cyPAPI`, set the required environment variables:
```bash
export PAPI_PATH=/global/homes/<first initial of your username>/<perlmutter-username>
export PAPI_DIR=/global/homes/<first initial of your username>/<perlmutter-username>
export PAPI_CUDA_ROOT=$CUDA_HOME
```
Optionally - you can also place:

```bash
export PAPI_CUDA_ROOT=$CRAY_CUDATOOLKIT_PREFIX 
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
Skip this step if you are using NOT using the papi module under perlmutter
2. Modify `torch_cuda.py`:
    - Set `unit = "cuda_7011"`
    - Change instances using `unit` to `"cuda"`:
    ```python
    matrix_A = torch.rand(m_rows, n_cols, device="cuda")
    matrix_B = torch.rand(m_rows, n_cols, device="cuda")
    ```

3. Run the test files:
    ```bash
    ./torch_cuda.py
    ./realtime.py
    ```

## Reason of installing papi rather than using the papi module on Perlmutter
We found that perftools-base causes an overhead issue when testing Treece code testing hardware counts making concerns for potential errors in collecting data.  Using the papi github repo helped show correct results.  In addition the reset() function to reset hardware counts was not correctly outputting the hardware counts for fmaa.

## Troubleshooting

Refer to the [NERSC documentation](https://docs.nersc.gov/environment/#:~:text=Many%20users%20include%20module%20load%20statements%20in%20their%20~/.bashrc%20to%20customize%20their%20startup%20modules%2C%20but%20this%20can%20cause%20unexpected%20side%2Deffects%20when%20loading%20other%20modules) for potential issues with environment variables.

If encountering an error such as Exception: PAPI Error -14: PAPI_start failed , make sure the dcgmi profile is paused. Optionally you can use the command:
```python 
python3 -m pdb torch_cuda.py
```

##PyTorch
Make sure Pytorch is installed under module load python in perlmutter.  If missing functions or modules exist, try installing pytorch with pip in your python environment

### Papi installation troubleshooting
If papi does not generate papi_avail under bin, try steps again and double check PAPI_PATH, PAPI_DIR enviromental variables.  I found that placing quotes do not set up paths right to your papi installation.  Or  `make clean` and retry installation steps.  Keep in mind I found that enviormenetal varaibels DO NOT save after loging out perlmutter session nor being recognized under a .bashrc file.  If there is a workaround saving these variables rather than retyping again, I would be happy to know!

To debug the file line by line. Refer to the documentation how to use pdb https://docs.python.org/3/library/pdb.html 

## License

cyPAPI is licensed under the [BSD 2-Clause License](LICENSE).





