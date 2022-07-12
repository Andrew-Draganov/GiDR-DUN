import numpy as np
from setuptools import setup
from distutils.core import Extension

setup(
    name='GradientDR',
    version='0.1.2.3',
    description='A package for evaluating dimensionality reduction algorithms',
    author='Andrew Draganov',
    author_email='draganovandrew@cs.au.dk',
    packages=[
        'GDR',
        'GDR.optimizer',
        'GDR.optimizer.numba_optimizers',
        'GDR.nndescent',
        'GDR.nndescent.py_files',
        'GDR.experiment_utils',
        'GDR.scripts'
    ],
    package_dir={'GDR': 'GDR'},
    extras_require={'pytorch': 'torch'},
    include_dirs=[
        np.get_include(),
    ],
    install_requires=[
        'cython',
        'matplotlib',
        'python-mnist',
        'numpy==1.21',
        'pandas',
        'pytest',
        'sklearn',
        'scipy',
        'tqdm',
        'umap-learn',
    ],
)
