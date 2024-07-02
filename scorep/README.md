# README.md

This file provides a guide for setting up Score-P version 8.4 and CudeGUI version 4.8.2 on NERSC's Perlmutter. 
- Score-P is a software tool designed for performance analysis of HPC applications. The steps below are slightly altered due to some issues we found during our initial process which followed the guide given at https://github.com/score-p/scorep_binding_python/wiki.
- Cube, which is used as performance report explorer for Scalasca and Score-P, is a generic tool for displaying a multi-dimensional performance space consisting of the dimensions:
  - performance metric
  - call path
  - system resource

## Table of Contents

- [Prerequisites](#prerequisites)
  - [Building and using mpi4py on Perlmutter](#building-and-using-mpi4py-on-perlmutter)
- [Score-P Binding Python Setup](#score-p-binding-python-setup)
  - [Configure and Build Score-P](#configure-and-build-score-p)
  - [Installing the Score-P Python Bindings](#installing-the-score-p-python-bindings)

## Prerequisites

A Conda environment was built to work with mpi4py on Perlmutter. These steps are outlined in NERSC's "Using Python at NERSC and Best Practices" YouTube training video.
- [Using Python at NERSC and Best Practices](https://www.youtube.com/watch?v=rPlHLUnGVtA&t=637s)
- Conda environment build begins at 11:14.

### Building and using mpi4py on Perlmutter

mpi4py provides a Python interface to MPI. mpi4py is available via:

```bash
module load Python
```
To install mpi4py with CUDA support in cray-mpich:

```bash
module load PrgEnv-gnu cudatoolkit craype-accel-nvidia80 conda
conda create -n mpi4py-gpu python=3.11 -y
conda activate mpi4py-gpu
MPICC="cc -shared" pip3 install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

Python binding issues were found during our initial configuration. We resolved the issue by installing:

```bash
conda install six
```

## Score-P Binding Python Setup
### Configure and Build Score-P

Download Score-P 8.4:

```bash
wget https://perftools.pages.jsc.fz-juelich.de/cicd/cubegui/branches/master/sources.08ac1e34.tar.gz
tar -xf scorep-8.4.tar.gz
cd scorep-8.4/
mkdir _build
cd _build
```

Configure and Install Score-P:

There was trouble finding libbfd and libunwind libraries during the initial configuration process. Configuration would halt due to unknown libraries not being found. We resolved the issues by including the download of both libraries in our configuration step:
- We specify the path where we want our scorep 8.4 to be installed by setting ‘--prefix’ equal to that desired path.
  - In our case, the desired installation path is --> ‘$HOME/scorep’
- We tell the configuration to download both the ‘libbfd’ and ‘libunwind’ libraries.
- Configuration can take 5-10 minutes.

```bash
../configure –enable-shared –prefix=$HOME/scorep –with-libunwind=download –with-libbfd=download
```

This allowed the configuration to be complete, but we still ran into more issues further down the line when running the make. To resolve these issues we needed to reconfigure our score-p and a couple of enviornment variables.
- Libraries should be installed in a vendor folder located in your prefix directory.
  - In our case the location was --> '$HOME/scorep/vendor/libunwind' and '$HOME/scorep/vendor/libbfd'
- Modified environment variables LD_LIBRARY_PATH and LIBRARY_PATH by specifying the path to the 'lib' directory in both libunwind and libbfd.

```bash
../configure –enable-shared –prefix=$HOME/scorep –with-libunwind=/path-to-scorep/vendor/libunwind –with-libbfd=/path-to-scorep/vendor/
libbfd
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path-to-scorep/vendor/libunwind/lib:/path-to-scorep/vendor/libbfd/lib
export LIBRARY_PATH=$LIBRARY_PATH:/path-to-scorep/vendor/libunwind/lib:/path-to-scorep/vendor/libbfd/lib
```

Lastly, we ran the build for score-p which was successful after all these minor alterations were made to the initial installation process.
- Build can take 35-45 minutes.

```bash
make 
make install
```

### Installing the Score-P Python Bindings

Before installing python bindings we need to specify our scorep path to avoid errors during the installation process:

```bash
export PATH=$PATH:/path-to-scorep/bin
```

To install the python bindings you simply need to clone the repository, and install it using pip

```bash
git clone https://github.com/score-p/scorep_binding_python
cd scorep_python_bindings/
pip3 install .
```

Score-P Python Bindings is now ready to use.
