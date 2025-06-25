import numpy
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


ext_modules = cythonize([
    Extension(
        "Pythonreflectivity",
        sources=["libs/Pythonreflectivity.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        extra_link_args=['-O3'],
        language='c++'
    ),
], compiler_directives={'language_level': "3"})

setup(
    ext_modules=ext_modules,
    packages=find_packages(where='libs'),
    package_dir={'': 'libs'},
)
