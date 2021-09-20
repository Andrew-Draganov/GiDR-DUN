from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('bbox', ['umap/_barnes_hut.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))
