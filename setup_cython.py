from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import os
import numpy

# os.environ['CC']='/usr/local/Cellar/gcc/11.2.0_3/bin/gcc-11'
# os.environ['CXX']='/usr/local/Cellar/gcc/11.2.0_3/bin/g++-11'

uniform_umap_build = Extension(
    'uniform_umap_opt',
    ['cython/uniform_umap.pyx'],
    language=['c'],

    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

umap_build = Extension(
    'umap_opt',
    ['cython/umap.pyx'],
    language=['c'],

    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

tsne_build = Extension(
    'tsne_opt',
    ['cython/tsne.pyx'],
    language=['c'],

    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)
CySetup(
    name='cython_dim_reduction',
    ext_modules=cythonize([uniform_umap_build, umap_build, tsne_build])
)
