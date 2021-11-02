from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize
import numpy

package = Extension(
    'barnes_hut',
    ['_barnes_hut.pyx'],
    include_dirs=[numpy.get_include()]
)
CySetup(ext_modules=cythonize([package]))
