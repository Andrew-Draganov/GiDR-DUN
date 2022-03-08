from distutils.core import setup as CySetup
from distutils.core import Extension
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize#, build_ext
import numpy
import os

# os.environ['CC']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin/nvcc'
# # os.environ['CXX']='/usr/local/Cellar/gcc/11.2.0_3/bin/g++-11'
#
# optimize = Extension(
#     'optimize',
#     ['cython/optimize.pyx'],
#     language=['c'],
#
#     compile_args = "-fPIC -stdpar -gpu=nordc -std=c++17",
#     link_args = "-shared -stdpar",
#     self.compiler.set_executable(
#         "compiler_so",
#         NVCPP_EXE + " " + compile_args
#     )
# self.compiler.set_executable("compiler_cxx", NVCPP_EXE)
# self.compiler.set_executable(
#     "linker_so",
#     NVCPP_EXE + " " + link_args
# )
#     include_dirs=[numpy.get_include()]
# )
#
# # optimize_frob = Extension(
# #     'optimize_frob',
# #     ['cython/optimize_frob.pyx'],
# #     language=['c'],
# #
# #     extra_compile_args=['-O3', '-march=native', '-ffast-math'],
# #     extra_link_args=[],
# #     include_dirs=[numpy.get_include()]
# # )
# CySetup(
#     name='cython_dim_reduction',
#     ext_modules=cythonize([optimize])#, optimize_frob])
# )


include_dirs = [numpy.get_include()]
library_dirs = []
libraries = []

NVCPP_EXE = 'nvcc' #os.environ.get("nvcc", None)
# NVCPP_EXE = CC if CC is not None and CC.endswith("nvcc") else None

if NVCPP_EXE is not None:
    NVCPP_HOME = os.path.dirname(os.path.dirname(NVCPP_EXE))
    include_dirs += [
        os.path.join(NVCPP_HOME, "include-stdpar")
    ]
    library_dirs += [
        os.path.join(NVCPP_HOME, "lib")
    ]


# noinspection PyPep8Naming
class custom_build_ext(build_ext):
    def build_extensions(self):
        if NVCPP_EXE:
            # Override the compiler executables. Importantly, this
            # removes the "default" compiler flags that would
            # otherwise get passed on to nvc++, i.e.,
            # distutils.sysconfig.get_var("CFLAGS"). nvc++
            # does not support all of those "default" flags
            compile_args = "-fPIC -stdpar -gpu=nordc -std=c++17"
            link_args = "-shared -stdpar"
            self.compiler.set_executable(
                "compiler_so",
                NVCPP_EXE + " " + compile_args
            )
            self.compiler.set_executable("compiler_cxx", NVCPP_EXE)
            self.compiler.set_executable(
                "linker_so",
                NVCPP_EXE + " " + link_args
            )
        build_ext.build_extensions(self)


ext = cythonize([
    Extension(
        '*',
        sources=['cython/*.pyx'],
        libraries=libraries,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        extra_compile_args=["-std=c++17"]
    )])

setup(
    name='cppsort',
    author='Ashwin Srinath',
    version='0.1',
    ext_modules=ext,
    zip_safe=False,
    cmdclass={'build_ext': custom_build_ext}
)