from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import numpy


optimize = Extension(
    'optimize',
    ['optimize.pyx'],
    libraries=['m'],

    # If compiling with clang, uncomment these and use
    # CC=/usr/local/opt/llvm/bin/clang++ python setup_cython.py install
    extra_compile_args=['-ffast-math', '-fopenmp'],
    extra_link_args=['-lomp'],

    # If compiling regularly, just use this
    # extra_compile_args=['-ffast-math'],

    include_dirs=[numpy.get_include()]
)

optimize_frob = Extension(
    'optimize_frob',
    ['optimize_frob.pyx'],
    libraries=['m'],

    # If compiling with clang, uncomment these and use
    # CC=/usr/local/opt/llvm/bin/clang++ python setup_cython.py install
    extra_compile_args=['-ffast-math', '-fopenmp'],
    extra_link_args=['-lomp'],

    # If compiling regularly, just use this
    # extra_compile_args=['-ffast-math'],

    include_dirs=[numpy.get_include()]
)
CySetup(
    name='cython_dim_reduction',
    ext_modules=cythonize([optimize, optimize_frob])
)
