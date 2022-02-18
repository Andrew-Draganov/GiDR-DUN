from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import os
import numpy

os.environ['CC']='/usr/local/Cellar/gcc/11.2.0_3/bin/gcc-11'
os.environ['CXX']='/usr/local/Cellar/gcc/11.2.0_3/bin/g++-11'
# os.environ['CC']='/usr/local/opt/llvm/bin/clang++'
# os.environ['CXX']='/usr/local/opt/llvm/bin/clang++'

# optimize = Extension(
#     'optimize',
#     ['optimize.pyx'],
#     libraries=['m'],
# 
#     # If compiling with clang, uncomment these and use
#     # CC=/usr/local/opt/llvm/bin/clang++ python setup_cython.py install
#     extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
#     extra_link_args=['-lomp', '-fopenmp'],
# 
#     # If compiling regularly, just use this
#     # extra_compile_args=['-ffast-math'],
# 
#     include_dirs=[numpy.get_include()]
# )

# if os.path.exists('optimize_frob.cpython-39-darwin.so'):
#     print('Removing optimize_frob.cpython-39-darwin.so \n')
#     os.remove('optimize_frob.cpython-39-darwin.so')
# if os.path.exists('optimize_frob.c'):
#     print('Removing optimize_frob.c \n')
#     os.remove('optimize_frob.c')

optimize_frob = Extension(
    'optimize_frob',
    ['optimize_frob.pyx'],
    # libraries=['m'],
    language=['c'],

    # If compiling with clang, uncomment these and use
    # CC=/usr/local/opt/llvm/bin/clang++ python setup_cython.py install
    extra_compile_args=['-O3', '-march=native', '-ffast-math'],
    # extra_compile_args=[],
    extra_link_args=['-fopenmp'],

    # If compiling regularly, just use this
    # extra_compile_args=['-ffast-math'],

    include_dirs=[numpy.get_include()]
)
CySetup(
    name='cython_dim_reduction',
    # ext_modules=cythonize([optimize, optimize_frob])
    ext_modules=cythonize([optimize_frob])
)
