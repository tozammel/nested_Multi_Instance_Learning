from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_info

ext_modules = [Extension('_hmmc', ['_hmmc.pyx'],
                         extra_compile_args=["-O3"], **get_info("npymath"))]

setup(name='_hmmc', cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
