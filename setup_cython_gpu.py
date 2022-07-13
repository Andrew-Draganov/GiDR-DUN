from distutils.core import setup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import numpy
import os

# GPU INSTALLATION REQUIRES FILLING IN THE BELOW VARIABLE!
# Example:
CUDA_PATH = '/usr/local/cuda-11/targets/x86_64-linux/include/'
# CUDA_PATH = None
# If not CUDA_PATH:
#     raise ValueError("Must supply cuda path for the cython gpu compilation")
# FIXME -- should get path automatically
os.environ['CXX']='/usr/bin/clang++'
os.environ['C++FLAGS'] = '-O3 -march=native -ffast-math'

gpu_opt = Extension(
    'gpu_gdr',
    ['GDR/cython/cython_files/gpu_gdr.pyx'],
    libraries=['gpu_dim_reduction'],
    library_dirs=['.'],
    language='c++',
    include_dirs=[
        numpy.get_include(),
        CUDA_PATH
    ],
    runtime_library_dirs=['.']
)

gpu_graph_build = Extension(
    'gpu_graph_build',
    ['GDR/cython/cython_files/gpu_graph_build.pyx'],
    libraries=['gpu_graph_weights'],
    library_dirs=['.'],
    language='c++',
    include_dirs=[
        numpy.get_include(),
        CUDA_PATH
    ],
    runtime_library_dirs=['.']
)

setup(
    name='gpu_gdr',
    ext_modules=cythonize([gpu_opt, gpu_graph_build])
)
