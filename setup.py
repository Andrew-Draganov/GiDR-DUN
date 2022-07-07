from setuptools import setup

setup(
    name='GradientDimReduction',
    version='0.1.0',
    description='A package for evaluating dimensionality reduction algorithms',
    author='Andrew Draganov',
    author_email='draganovandrew@cs.au.dk',
    packages=['nndescent'],
    package_dir={'nndescent': 'nndescent'},
    extras_require={'pytorch': 'torch'},
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
