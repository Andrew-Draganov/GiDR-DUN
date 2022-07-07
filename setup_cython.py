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
os.environ['CC']='/usr/bin/gcc'
os.environ['CFLAGS'] = '-fopenmp -O3 -march=native -ffast-math'
# os.environ['CXX']='/usr/bin/gcc++'

gidr_dun_build = Extension(
    'gidr_dun_opt',
    ['GDR/cython/cython_files/gidr_dun.pyx'],
    language=['c'],

    # compile_args=,
    extra_link_args=[],
    include_dirs=[
        numpy.get_include(),
    ],
)

umap_build = Extension(
    'umap_opt',
    ['GDR/cython/cython_files/umap.pyx'],
    language=['c'],

    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

tsne_build = Extension(
    'tsne_opt',
    ['GDR/cython/cython_files/tsne.pyx'],
    language=['c'],

    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)
CySetup(
    name='cython_dim_reduction',
    ext_modules=cythonize([gidr_dun_build, umap_build, tsne_build])
)
