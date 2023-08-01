from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "nhsolver",
        sources=["MCsolver.cc","nhsolver.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
        include_dirs=[numpy.get_include()],
        extra_link_args=["-larmadillo"],
    )
]

setup(
    name="nhsolver",
    ext_modules=cythonize(ext_modules),
)
