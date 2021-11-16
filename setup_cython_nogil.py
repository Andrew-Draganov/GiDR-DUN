from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize, build_ext
import numpy


optimize = Extension(
    'optimize',
    ['optimize_nogil.pyx'],
    libraries=['m'],

    # If compiling with clang, uncomment these and use
    # CC=/usr/local/opt/llvm/bin/clang python setup_cython.py install
    #   - assumes you've installed llvm with `brew install llvm`
    extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
    extra_link_args=['-lomp', '-fopenmp'],

    # If compiling regularly, just use this
    # extra_compile_args=['-ffast-math'],

    include_dirs=[numpy.get_include()]
)
CySetup(
    name='cython_dim_reduction',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([optimize])
)
