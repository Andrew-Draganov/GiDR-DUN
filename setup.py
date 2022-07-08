import numpy as np
from setuptools import setup
from distutils.core import Extension

setup(
    name='GradientDimReduction',
    version='0.1.0',
    description='A package for evaluating dimensionality reduction algorithms',
    author='Andrew Draganov',
    author_email='draganovandrew@cs.au.dk',
    packages=['nndescent'],
    package_dir={'nndescent': 'nndescent'},
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
        'sklearn',
        'scipy',
        'tqdm',
        'umap-learn',
    ],
)
