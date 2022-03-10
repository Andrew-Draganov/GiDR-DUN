from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import numpy

optimize_gpu = Extension(
    'optimize_gpu',
    ['cython/optimize.pyx'],
    libraries=['gpu_dim_reduction'],
    library_dirs=['cython', '.', '..'],
    language='c++'
)
CySetup(
    name='cython_dim_reduction',
    ext_modules=cythonize([optimize_gpu])
)
