from distutils.core import setup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import os
import numpy

# MAC INSTALLATION
# ----------------
#
# If you are installing on mac, you need to install a version of LLVM with openmp
# You can then link it by uncommenting the lines below and filling in the path
# to your version of LLVM
#
# os.environ['CC']='/path/to/llvm/gcc'
# os.environ['CXX']='/path/to/llvm/g++'

# Note, this does not work on the M1 chip

# Example below
# -------------
os.environ['CC']='/usr/bin/clang++'
# os.environ['CXX']='/usr/bin/clang++'

gdr_build = Extension(
    'gdr_cython',
    ['GDR/cython/cython_files/gdr_cython.pyx', 'GDR/cython/utils/cython_utils.cpp'],
    language=['c++'],
    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math', '-std=c++0x'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()],
)

umap_build = Extension(
    'umap_cython',
    ['GDR/cython/cython_files/umap_cython.pyx'],
    language=['c++'],
    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math', '-std=c++0x'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

tsne_build = Extension(
    'tsne_cython',
    ['GDR/cython/cython_files/tsne_cython.pyx'],
    language=['c++'],
    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math', '-std=c++0x'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)
setup(
    name='cython_gdr',
    ext_modules=cythonize([gdr_build, umap_build, tsne_build])
)
