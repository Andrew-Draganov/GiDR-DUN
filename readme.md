## Gradient Dimensionality Reduction Differences and Unification Library
Written by Andrew Draganov, with help from Jakob Rødsgaard Jørgensen and Katrine Scheel Nellemann.

This library has the following implementations of UMAP, TSNE, and Uniform UMAP
 - Numba \-\- UMAP, Uniform UMAP
 - Cython \-\- UMAP, TSNE, Uniform UMAP
 - Cuda \-\- Uniform UMAP

You can test each of the above by running the `dim_reduce_dataset.py` script. Command-line params
will dictate whether you run the numba, cython, or GPU implementation. We list some examples
for calling these below:
 - To run TSNE with Cython, call `python dim_reduce_dataset.py --optimize-method tsne --normalized`
 - To run Uniform UMAP in numba, call `python dim_reduce_dataset.py --optimize-method umap --numba --sym-attraction`
 - To run on the GPU, call `python dim_reduce_dataset.py --gpu`
The script defaults to running Uniform UMAP in Cython on the MNIST dataset.

### Installation

All installs begin with installing the base `setup.py` file from the home directory.

After this, you can compile the cython code by calling `python setup_cython.py install`. This requires
a compiler with `OpenMP`, which is a default on Linux machines. For Mac, you first need to install a
version of `LLVM` with `OpenMP` and perform the compilation with this. This can simply be done
by calling `brew install llvm`. These can be defined in the `setup_cython.py` file in the commented area.

Note, this does not work on the Mac M1, as these run on ARM chips.

### Hyperparameter Testing

Part of the motivation for making an independent library to run TSNE and UMAP was to test all
of the relevant hyperparameters. These can be evaluated using the other command-line parameters
in `dim_reduce_dataset.py`. All experiments in the paper can be reproduced using the
`run_analysis.py` script.

Some specific hyperparameter experiment examples can be found below:
 - To run TSNE with the Frobenius norm and UMAP's normalization, call
   `python dim_reduce_dataset.py --optimize-method tsne --frob`
 - To run UMAP with all of TSNE's params except the normalization, call
   `python dim_reduce_dataset.py --optimize-method umap --tsne-symmetrization --random-init
    --tsne-scalars 
