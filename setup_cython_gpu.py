from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import numpy

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
    # FIXME -- this should find the cuda path programmatically
    include_dirs=[
        numpy.get_include(),
        '/usr/local/cuda-11/targets/x86_64-linux/include/'
    ],
    runtime_library_dirs=['.']
)
CySetup(
    name='cython_dim_reduction',
    ext_modules=cythonize([optimize_gpu])
)
