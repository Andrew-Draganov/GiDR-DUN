from setuptools import setup

setup(
    name='Dimensionality Reduction Analysis',
    version='0.1.0',
    description='A package for evaluating dimensionality reduction algorithms',
    author='Andrew Draganov',
    author_email='draganovandrew@cs.au.dk',
    install_requires=[
        'cython',
        'matplotlib',
        'numpy==1.21',
        'numba',
        'pynndescent',
        'sklearn',
        'scipy',
        'tensorflow',
        'tensorflow_datasets'
    ],
)
