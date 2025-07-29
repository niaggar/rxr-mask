# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("../../rxrmask"))

project = 'rxrmask'
copyright = '2025, Nicolas Aguilera G'
author = 'Nicolas Aguilera G'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",            # extrae docstrings
    "sphinx.ext.napoleon",           # soporta NumPy/Google style
    "sphinx_autodoc_typehints",      # incluye hints de tipo
    "sphinx.ext.viewcode",           # link al código fuente
]

templates_path = ['_templates']
exclude_patterns = []

sphinx_gallery_conf = {
    "examples_dirs": "../examples",   # donde pongas tus ejemplos
    "gallery_dirs": "examples",     # dónde generar las páginas
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
