from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import numpy

_barnes_hut = Extension(
    'barnes_hut',
    ['_barnes_hut.pyx'],
    libraries=['m'],
    # extra_compile_args=['-ffast-math', '-fopenmp'],
    # extra_link_args=['-lomp'],
    extra_compile_args=['-ffast-math'],
    include_dirs=[numpy.get_include()]
)
CySetup(
    name='cython_dim_reduction',
    ext_modules=cythonize([_barnes_hut])
)
