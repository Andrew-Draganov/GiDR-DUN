# "GiDR-DUN: Gradient Dimensionality Reduction Differences and Unification" Library
Written by Andrew Draganov, with help from Jakob Rødsgaard Jørgensen and Katrine Scheel Nellemann.

## Overview

This library has the following implementations of UMAP, TSNE, and GiDR\_DUN
 - Numba \-\- UMAP, GiDR\_DUN
 - Cython \-\- UMAP, TSNE, GiDR\_DUN
 - Cuda \-\- GiDR\_DUN
 - Pytorch \-\- GiDR\_DUN

You can test each of the above by running the `dim_reduce_dataset.py` script. Command-line params
will dictate whether you run the numba, cython, or GPU implementation. We list some examples
for calling these below:
 - To run our implementation of TSNE with Cython, call `python dim_reduce_dataset.py --optimize-method tsne --normalized`
 - To run our implementation of UMAP in cython, call `python dim_reduce_dataset.py --optimize-method umap --sym-attraction`
 - To instead run our implementation of UMAP in numba, call `python dim_reduce_dataset.py --optimize-method umap --numba --sym-attraction`
 - To run on the GPU, call `python dim_reduce_dataset.py --gpu`
 - To run on the GPU with pytorch, call `python dim_reduce_dataset.py --gpu --torch`
Further examples are listed in the `run_cpu_test` make target, which performs several quick experiments.

The script defaults to running GiDR\_DUN in Cython on the MNIST dataset.

## Installation

We suggest using the targets in the attached `makefile`. The steps are as follows:
 - Make sure conda is installed
 - Create the conda environment using `make create_BLANK_env`. Your options are creating a python, cuda, or pytorch environment.
     - For the regular python environment, call `make create_python_env`
     - For the cuda environment, call `make create_rapids_env`
     - For the torch environment, call `make create_torch_env`
 - Enter into the conda environment you made. These commands should respectively be:
     - `conda activate GiDR_DUN`
     - `conda activate GiDR_DUN_rapids`
     - `conda activate GiDR_DUN_torch`
 - We now install the relevant libraries and compile the C code:
     - `make insall_python_env` will allow you to run the numba and torch optimizations
     - `make insall_cython_env` will allow you to do the default cython optimizations
     - `make insall_cuda_code` will install the cuda wrappers for the gpu implementation
 - If you have installed the cython code, you can check that everything works by calling `make run_cpu_test`
     - Similarly for cuda code, `make run_gpu_test`
 - You can then remake the plots from the paper by `make run_analysis` and `make run_gpu_analysis`

If you intend to only run the basic numba implementations, then it is sufficient to just pip install the setup.py file.
This requires that you add the `--numba` flag/parameter when invoking GiDR\_DUN. Note that this does NOT implement
the original TSNE optimization protocol, as the Barnes\_ Hut tree data structure cannot be re-made in numba.
However, you can use GiDR\_DUN to obtain TSNE embeddings by adding the `--optimize-method tsne` flag.

## Hyperparameter Testing

Part of the motivation for making an independent library to run TSNE and UMAP was to test all
of the relevant hyperparameters. These can be evaluated using command-line parameters
in `dim_reduce_dataset.py`.

Some specific hyperparameter experiment examples:
 - To run TSNE with the Frobenius norm and UMAP's normalization, call
   `python dim_reduce_dataset.py --optimize-method tsne --frob`
 - To run UMAP with all of TSNE's params except the normalization, call
   `python dim_reduce_dataset.py --optimize-method umap --tsne-symmetrization --random-init
    --tsne-scalars`

##
Contact -- for questions please raise an issue or (if you want a response) email draganovandrew@cs.au.dk
