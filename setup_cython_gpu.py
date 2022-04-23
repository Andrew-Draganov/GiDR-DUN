from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import numpy

# GPU INSTALLATION REQUIRES FILLING IN THE BELOW VARIABLE!
# Example:
# CUDA_PATH = '/usr/local/cuda-11/targets/x86_64-linux/include/'
CUDA_PATH = '/usr/local/cuda-11/targets/x86_64-linux/include/'
# If not CUDA_PATH:
#     raise ValueError("Must supply cuda path for the cython gpu compilation")

optimize_gpu = Extension(
    'optimize_gpu',
    ['cython/gpu_uniform_umap.pyx'],
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
CySetup(
    name='cython_dim_reduction_gpu',
    ext_modules=cythonize([optimize_gpu])
)
