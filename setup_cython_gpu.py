from distutils.core import setup as CySetup
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
os.environ['CXX']='/usr/bin/gcc'
os.environ['C++FLAGS'] = '-fopenmp -O3 -march=native -ffast-math'

optimize_gpu = Extension(
    'optimize_gpu',
    ['GDR/cython/cython_files/gpu_gidr_dun.pyx'],
    libraries=['gpu_dim_reduction'],
    library_dirs=[
        'cython',
        '.',
        '..'
    ],
    language='c++',
    include_dirs=[
        numpy.get_include(),
        CUDA_PATH
    ],
    runtime_library_dirs=['.']
)

graph_weights_build = Extension(
    'graph_weights_build',
    ['GDR/cython/cython_files/graph_weights.pyx'],
    libraries=['gpu_graph_weights'],
    library_dirs=[
        'cython',
        '.',
        '..'
    ],
    language='c++',
    include_dirs=[
        numpy.get_include(),
        CUDA_PATH
    ],
    runtime_library_dirs=['.']
)

CySetup(
    name='cython_dim_reduction_gpu',
    ext_modules=cythonize([optimize_gpu, graph_weights_build])
)
