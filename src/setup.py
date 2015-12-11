from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("quilting_parallel", ["quilting_parallel.pyx"], include_dir=[np.get_include()])]

setup(
	name = "quilting_parallel",
	cmdclass = {'build_ext': build_ext},
	include_dirs = [np.get_include()],
	ext_modules = ext_modules
)