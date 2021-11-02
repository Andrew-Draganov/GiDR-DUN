from distutils.core import setup as CySetup
from distutils.core import Extension
from setuptools import setup
from Cython.Build import cythonize
import numpy

PySetup(
    name='Dimensionality Reduction Analysis',
    version='0.1.0',
    description='A package for evaluating dimensionality reduction algorithms',
    author='Andrew Draganov',
    author_email='draganovandrew@cs.au.dk',
    packages=[
        'numpy',
        'numba',
        'sklearn',
        'pyximport',
        'scipy',
        'matplotlib',
        'tensorflow',
    ],
    install_requires=['cython']
)

package = Extension(
    'barnes_hut',
    ['_barnes_hut.pyx'],
    include_dirs=[numpy.get_include()]
)
CySetup(ext_modules=cythonize([package]))
