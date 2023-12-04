# Gradient Dimensionality Reduction; an implementation of tSNE, UMAP, and GDR
Corresponds to the paper "ActUp: Analyzing and Consolidating tSNE and UMAP".

The name of the repository corresponds to when we wanted to use the acronym GiDR-DUN: **G**radient **D**imensionality **R**eduction; **D**ifferences between and **U**nification of tSNE and UMAP.
Written by Andrew Draganov, Jakob Rødsgaard Jørgensen and Katrine Scheel Nellemann.

## Overview

This library contains simplified and standalone implementations of TSNE and UMAP. We also include our generalization of them, which can recreate either one by flipping the normalization.
Furthermore, we support multiple backends. We have the following implementations available:
  - UMAP
    - numba
    - cython
  - TSNE
    - cython
  - GDR (which can recreate both TSNE and UMAP embeddings)
    - numba
    - cython
    - gpu
    - pytorch (cpu/gpu)
    
On our machine, the cython implementation performs the gradient updates almost an order of magnitude faster than the numpy one. The gpu code is simply a barebones translation of the cython code into CUDA.

We have tried to trim all of the fat possible out of these algorithms. This will hopefully make the methods easier to extend
for future research/design. This means that some features are missing, such as UMAP's ability to call `.fit()` and then `.transform()`
(we only support `.fit_transform()`). Additionally, we default to the NNdescent nearest neighbors algorithm in all cases.

Despite this, we added the option to toggle all of the hyperparameters between the UMAP and TSNE algorithms.
For example, you can run TSNE with UMAP's pseudo-distance metric and normalization.
Or UMAP with TSNE's symmetrization and the standard Euclidean distance metric. etc. etc.

## Installation

### Numba install
If you'd like to simply install the numba versions of UMAP and GDR, then you are good to go with a simple

    pip install GradientDR

You can then use it by calling

    from GDR import GradientDR
    dr = GradientDR()
    dr.fit_transform(dataset)

### Cython install
Cython requires OpenMP support. We must compile it against clang++ as gcc++ does not handle memory allocation the correct way
when parallelizing.
  - Clone the repository and `cd` into it
  - Export your version of clang++ that has OpenMP support to the environment variable `CLANG_PATH`.
    - `export CLANG_PATH=clang++` should work on linux machines
    - Mac users must first install llvm with OpenMP support and set the `CLANG_PATH` variable to point to that clang++ compiler
  - Run `python setup_cython.py build_ext --inplace` from the base directory in a python>=3.8 venv or conda environment

Run the cython implementations by

    from GDR import GradientDR
    dr = GradientDR(cython=True)
    dr.fit_transform(dataset)

### GPU install
  - Clone the repository and `cd` into it
  - Run `make install_cuda_code` from the base directory in a python>=3.8 venv or conda environment
    - We have tested for cuda 11, 11.3, 11.5, and 11.6.
    - This assumes that you have a default `nvcc` version. If this environment variable is not set, you must first export it
        - `export nvcc=/usr/local/cuda-11.X/bin/nvcc`

Run the gpu implementations by

    from GDR import GradientDR
    dr = GradientDR(gpu=True)
    dr.fit_transform(dataset)

### Usage
You can set up a model to run each algorithm by the following constructors:
  - UMAP -- `dr = GradientDR(optimize_method='umap')`
  - TSNE -- `dr = GradientDR(optimize_method='tsne', cython=True)`
    - Requires `cython=True` as TSNE's Barnes-Hut trees cannot be written into numba easily
  - GDR -- `dr = GradientDR(optimize_method='gdr')`
    - This is the default as it can reproduce both UMAP and TSNE by toggling the `normalized` parameter

##
Contact -- for questions please raise an issue or (if you want a response) email draganovandrew@cs.au.dk

If you use this code, please cite our paper :)
