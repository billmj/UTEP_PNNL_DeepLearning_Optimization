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
  - [Score-P Usage](#score-p-usage)
- [Cube GUI Installation](#cube-gui-installation)
  - [Download NoMachine](#download-nomachine)
  - [Create a new connection for Perlmutter within NoMachine](#create-a-new-connection-for-perlmutter-within-nomachine)
  - [Configure and Build Cube GUI](#configure-and-build-cube-gui)

## Prerequisites

A Conda environment was built to work with mpi4py on Perlmutter. These steps are outlined in NERSC's "[Using Python at NERSC and Best Practices](https://www.youtube.com/watch?v=rPlHLUnGVtA&t=637s)" YouTube training video.
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

Issues with Score-P and Cube GUI were found during our initial configuration. We resolved the issues by installing:

```bash
conda install six
conda install -c conda-forge qt=5
```

## Score-P Binding Python Setup
### Configure and Build Score-P

Build a scorep directory in your $HOME path and download [Score-P 8.4](https://www.vi-hps.org/projects/score-p):

```bash
mkdir scorep
wget https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/scorep-8.4/scorep-8.4.tar.gz
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
- Configuration can take 8-10 minutes.

```bash
../configure --enable-shared --prefix=$HOME/scorep --with-libunwind=download --with-libbfd=download
```

This allowed the configuration to be complete, but we still ran into more issues further down the line when running the make. To resolve these issues we needed to reconfigure our score-p and a couple of enviornment variables.
- Libraries should be installed in a vendor folder located in your prefix directory.
  - In our case the location was --> '$HOME/scorep/vendor/libunwind' and '$HOME/scorep/vendor/libbfd'
- Modified environment variables LD_LIBRARY_PATH and LIBRARY_PATH by specifying the path to the 'lib' directory in both libunwind and libbfd.

```bash
../configure --enable-shared –prefix=$HOME/scorep --with-libunwind=/path-to-scorep/vendor/libunwind --with-libbfd=/path-to-scorep/vendor/libbfd
libbfd
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path-to-scorep/vendor/libunwind/lib:/path-to-scorep/vendor/libbfd/lib
export LIBRARY_PATH=$LIBRARY_PATH:/path-to-scorep/vendor/libunwind/lib:/path-to-scorep/vendor/libbfd/lib
```

Lastly, we ran the build for score-p which was successful after all these minor alterations were made to the initial installation process.
- Build can take 40-45 minutes.

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
git clone https://github.com/score-p/scorep_binding_python.git
cd scorep_binding_python/
pip3 install .
```

Score-P Python Bindings is now ready to use.

## Score-P Usage

More information on score-p usage can be found on the [Score-P Binding Python GitHub](https://github.com/score-p/scorep_binding_python?tab=readme-ov-file#use) page.

### Invoking Score-P

You can profile your application using the python bindings that were just installed. To do this, allow the scorep python module execute your application.
- The testing script can be found within the test folder in the scorep binding python git directory that was cloned.
- Must be within this test folder when invoking scorep for profiling and tracing. If not, you may encounter a 'scorep._bindings' error when executing.

```bash
cd scorep_binding_python/test/
python3 -m scorep test_scorep.py
```

This will create a folder in the same directory, with a name simialr to:

```bash
scorep-20180514_1012_10320848076853/
```

## Cube GUI Installation
To visualize the profiling, generated by Score-P in Perlmutter, we needed to install [NoMachine/NX](https://docs.nersc.gov/connect/nx/) to get a Perlmutter GUI.
- NoMachine (formerly NX) is a computer program that handles remote X Window System connections and offers several performance improvements over traditional X11 forwarding.
- NoMachine can greatly improve the response time of X Windows and is the recommended method of interacting with GUIs and visualization tools running on NERSC resources.

### Download NoMachine
Download [NoMachine Enterprise Client](https://downloads.nomachine.com/download-enterprise/#NoMachine-Enterprise-Client) for your respective Machine's OS.
- We followed the Windows download installation process
- Opted not to go with the sshproxy setup

### Create a new connection for Perlmutter within NoMachine. 
Our connection setup was as follows:
- Machine Address
  - Name: Connection to NERSC
  - Host: nxcloud.nersc.gov
  - Port: 22
  - Protocol: SSH
- Machine Configuration
  - Authentication: Use password authentication
  - Since we are not using sshproxy we will be required to use our NERSC password and OTP every time we connect to NoMachine.

### Configure and Build Cube GUI 

Download [CubeGUI 4.8.2](https://www.scalasca.org/software/cube-4.x/download.html):

```bash
wget https://apps.fz-juelich.de/scalasca/releases/cube/4.8/dist/cubegui-4.8.2.tar.gz
tar -xf cubegui-4.8.2.tar.gz
cd cubegui-4.8.2/
mkdir _build
cd _build
```

Configure and Install Cube GUI:

We ran into issues with the 'cubelib-config' file not being found during the configuration process. We resolved issues by doing the following:
- We specify the path where we want our Cube GUI to be installed by setting ‘--prefix’ equal to that desired path.
  - In our case, the desired installation path is --> ‘$HOME/cubegui’
- We found our 'cubelib-config' file within the scorep directory that we built above.
  - It is located in the 'bin' directory within the '--prefix' that was entered in the Score-P configuration.
- Build can take 40-45 minutes.

```bash
../configure --prefix=$HOME/cubegui --with-cubelib=/path-to-scorep/bin
make 
make install
```

Lastly, be sure to add the path that you entered in '--prefix' to the $PATH environment variable.

```bash
export PATH=$PATH:/path-to-cubegui/bin
```
