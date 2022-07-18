from setuptools import setup
from distutils.core import Extension

setup(
    name='GradientDR',
    version='0.1.3',
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
    package_dir={'nndescent': 'GDR/nndescent', 'GDR': 'GDR'},
    extras_require={'pytorch': 'torch', 'umap': 'umap-learn==0.5.3'},
    install_requires=[
        'cython==0.29.30',
        'matplotlib==3.5.2',
        'python-mnist==0.7',
        'numpy==1.21',
        'numba==0.55.2',
        'pandas==1.4.3',
        'pytest==7.1.2',
        'scikit-learn==1.1.1',
        'scipy==1.8.1',
        'tqdm==4.64.0',
    ],
)
