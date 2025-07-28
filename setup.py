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

# ------------------------------------------------
# 3) Llamada a setup()
# ------------------------------------------------
setup(
    name="rxrmask",                        # nombre en PyPI
    version="0.1.0",
    description="Herramientas para modelar resonant X-ray reflectivity",
    long_description_content_type="text/markdown",
    author="Tu Nombre",
    author_email="tu@correo.com",
    url="https://github.com/tu_usuario/rxrmask",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    setup_requires=[
        "cython>=3.0.0",
        "numpy>=1.26",
    ],
    install_requires=[
        "numpy>=1.26",
        "scipy",
        "matplotlib",
        "pint",
    ],
    # packages puramente Python y tu extensión (detecta pythonreflectivity y rxrmask)
    packages=find_packages(exclude=["tests", "docs"]),
    include_package_data=True,
    # incluir el .pyx en el sdist para que pip lo compile
    package_data={
        "pythonreflectivity": ["Pythonreflectivity.pyx"],
    },
    ext_modules=ext_modules,
    # si tuvieras comandos de consola:
    # entry_points={
    #   "console_scripts": ["rxr-tool=rxrmask.cli:main"],
    # },
)
