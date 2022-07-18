from distutils.core import setup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import os
import sys
import numpy

try:
    os.environ['CXX'] = os.environ['CLANG_PATH']
    assert os.environ['CXX']
except:
    raise ValueError('Must provide a path to clang++ with openmp for cython to compile against\n'
                     '\tExample: `export CLANG_PATH=clang++`')

gdr_build = Extension(
    'gdr_cython',
    ['GDR/cython/cython_files/gdr_cython.pyx'],
    language=['c++'],
    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()],
)

umap_build = Extension(
    'umap_cython',
    ['GDR/cython/cython_files/umap_cython.pyx'],
    language=['c++'],
    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

tsne_build = Extension(
    'tsne_cython',
    ['GDR/cython/cython_files/tsne_cython.pyx'],
    language=['c++'],
    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)
setup(
    name='cython_gdr',
    ext_modules=cythonize([gdr_build, umap_build, tsne_build])
)
