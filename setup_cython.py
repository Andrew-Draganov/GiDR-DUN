from distutils.core import setup as CySetup
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

# Example below
# -------------
# os.environ['CC']='/usr/local/Cellar/gcc/11.2.0_3/bin/gcc-11'
# os.environ['CXX']='/usr/local/Cellar/gcc/11.2.0_3/bin/g++-11'

uniform_umap_build = Extension(
    'uniform_umap_opt',
    ['cython/cython_files/uniform_umap.pyx'],
    language=['c'],

    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[
        numpy.get_include(),
    ],
)

umap_build = Extension(
    'umap_opt',
    ['cython/cython_files/umap.pyx'],
    language=['c'],

    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

tsne_build = Extension(
    'tsne_opt',
    ['cython/cython_files/tsne.pyx'],
    language=['c'],

    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)
CySetup(
    name='cython_dim_reduction',
    ext_modules=cythonize([uniform_umap_build, umap_build, tsne_build])
)
