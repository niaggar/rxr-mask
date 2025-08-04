import io
import os
import numpy
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

ext_modules = cythonize(
    [
        Extension(
            "Pythonreflectivity",
            sources=[os.path.join("pythonreflectivity", "Pythonreflectivity.pyx")],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3"],
            extra_link_args=["-O3"],
            language="c++",
        ),
    ],
    compiler_directives={"language_level": "3"},
)

setup(
    name="rxrmask",
    version="0.1.0",
    description="Tools for modeling resonant X-ray reflectivity.",
    long_description_content_type="text/markdown",
    author="Nicolas Aguilera G.",
    author_email="niagar25@gmail.com",
    url="https://github.com/niaggar/rxr-mask",
    # license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    setup_requires=[
        "cython",
        "numpy",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pint",
        "joblib",
        "udkm1Dsim"
    ],
    packages=find_packages(exclude=["tests", "docs"]),
    include_package_data=True,
    package_data={
        "pythonreflectivity": ["Pythonreflectivity.pyx"],
        "rxrmask": [
            "form_factor/*",
            "magnetic_form_factor/*",
            "atomic_mass.txt",
        ],
    },
    ext_modules=ext_modules,
)
