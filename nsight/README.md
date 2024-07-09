NVIDIA Nsight includes both the nsys and ncu commands for collecting performance data. Both can be used with Python deep learning codes. nsys is used to collect time-based profiling data, and ncu can be used to collect hardware counter metrics. These tools collect data for low-level CUDA kernels.

To use these commands on Perlmutter, make sure that the cudatoolkit module is loaded. With either command, you can use --help to see the available options and arguments. NVIDIA also has documentation pages as follows:

Nsight Systems

https://developer.nvidia.com/nsight-systems

https://docs.nvidia.com/nsight-systems/

Nsight Compute

https://developer.nvidia.com/nsight-compute

https://docs.nvidia.com/nsight-compute/

Here is some Python/PyTorch-specific information, but some of it is out-of-date with respect to some options:

https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223

https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59

