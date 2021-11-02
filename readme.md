The main code to run the dimensionality comparison is at `umap/mnist_dr.py`.
Running it without command line parameters via `python umap/mnist_dr.py` will
simply apply base UMAP to the MNIST dataset. You can view command-line parameter
options with `python umap/minst_dr.py -h`.

I run it with Python 3.9.7. You can set up the environment by:

    brew install python
    # Go to the directory you want this file to be
    git clone git@github.com:Andrew-Draganov/probabilistic_dim_reduction.git
    cd probabilistic_dim_reduction
    # Make a virtual environment for python
    python3 -m venv dim_reduc_env
    # Enter the python virtual environment
    source dim_reduc_env/bin/activate

    # Verify that it is an appropriate python version
    # The below command should place you in a shell with Python 3.X
    python

    pip3 install cython numpy
    pip3 install .
