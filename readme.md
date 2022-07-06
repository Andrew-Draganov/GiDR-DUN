# "GiDR-DUN: Gradient Dimensionality Reduction Differences and Unification" Library
Written by Andrew Draganov, Jakob Rødsgaard Jørgensen and Katrine Scheel Nellemann.

## Overview

This library has the following implementations of UMAP, TSNE, and GiDR\_DUN
 - Numba \-\- UMAP, GiDR\_DUN
 - Cython \-\- UMAP, TSNE, GiDR\_DUN
 - Cuda \-\- GiDR\_DUN
 - Pytorch \-\- GiDR\_DUN

The script defaults to running GiDR\_DUN in Cython on the MNIST dataset. We note that GiDR\_DUN can recreate both TSNE and UMAP outputs at UMAP speeds.

## Installation

We suggest using the targets in the attached `makefile`. The steps are as follows:
 - Make sure conda is installed
 - Create the conda environment using one of the make targets. Your options are creating a python, cuda, or pytorch environment.
     - For the regular python environment, call `make create_python_env`
     - For the cuda environment, call `make create_rapids_env`
     - For the torch environment, call `make create_torch_env`
 - Enter into the conda environment you made. These commands should respectively be:
     - `conda activate GiDR_DUN`
     - `conda activate GiDR_DUN_rapids`
     - `conda activate GiDR_DUN_torch`
 - We now install the relevant libraries and compile the C code:
     - `make insall_python_env` will allow you to run the numba optimizations
     - `make insall_cython_env` will allow you to do the default cython optimizations as well as the numba ones
     - `make insall_cuda_code` will install the cuda wrappers for the gpu implementation as well as the cython and numba ones
         - **NOTE** we default to cuda 11.5 in the makefile. If you'd like to change this, changes must be made in the make target and the corresponding
`setup_cython_gpu.py` script.
     - If you'd like to run the pytorch GPU optimizer, simply enter the `GiDR_DUN_torch` conda environment and install the setup.py
 - If you have installed the cython code, you can check that everything works by calling `make run_cpu_test`
     - Similarly for cuda code, `make run_gpu_test`
 - You can then remake the plots from the paper by `make run_analysis` and `make run_gpu_analysis`

If you intend to only run the basic numba implementations, then it is sufficient to just pip install the setup.py file.
This requires that you add the `--numba` flag/parameter when invoking GiDR\_DUN. Note that this does NOT implement
the original TSNE optimization protocol, as the Barnes\_ Hut tree data structure cannot be re-made in numba.
However, you can use GiDR\_DUN to obtain TSNE embeddings by adding the `--optimize-method tsne` flag.

Note that on highly distributed systems, cython does a terrible job with the parallelization. We find that numba
is more consistent across server sizes but that cython outperforms numba on small systems.

## Hyperparameter Testing

Part of the motivation for making an independent library to run TSNE and UMAP was to test all
of the relevant hyperparameters. These can be evaluated using command-line parameters
in `dim_reduce_dataset.py`.

You can test each of the numba/cython/cuda/torch optimizers by running the `dim_reduce_dataset.py` script.
Command-line params dictate which optimizer you use and what the hyperparameter values are. We list some examples
for choosing the optimizers below:
 - To run our implementation of TSNE with Cython, call `python dim_reduce_dataset.py --optimize-method tsne --normalized`
 - To run our implementation of UMAP in cython, call `python dim_reduce_dataset.py --optimize-method umap --sym-attraction`
 - To instead run our implementation of UMAP in numba, call `python dim_reduce_dataset.py --optimize-method umap --numba --sym-attraction`
 - To run GiDR\_DUN on the GPU, call `python dim_reduce_dataset.py --gpu`
 - To run GiDR\_DUN on the GPU with pytorch, call `python dim_reduce_dataset.py --gpu --torch`

If you'd instead like to run the ORIGINAL umap-learn and sklearn umap and tsne, these can be chosen by the `--dr-algorithm`
flag to `dim_reduce_dataset.py`. Note that this overrides all other optimizer flags such as `--numba`, `--gpu`, and `--torch`.
For further clarity on how the algorithm gets loaded, refer to `experiment_utils/get_algorithm.py`.

Further examples are listed in the `run_cpu_test` make target, which performs several quick experiments.
Some specific hyperparameter experiment examples:
 - To run TSNE with the Frobenius norm and UMAP's normalization, call
   `python dim_reduce_dataset.py --optimize-method tsne --frob`
 - To run UMAP with all of TSNE's params except the normalization, call
   `python dim_reduce_dataset.py --optimize-method umap --tsne-symmetrization --random-init
    --tsne-scalars`

You can recreate the plots from the original paper by checking out commit 7cecfc6. The plots were made using
the `run_analysis.py` script. Plots are then made using the `experiment_utils/read_metrics.py` script.
I will unfortunately not maintain those scripts as they are big and gross and I want to make this into a
relatively clean library.

##
Contact -- for questions please raise an issue or (if you want a response) email draganovandrew@cs.au.dk
