rxrmask Documentation
=====================

rxrmask is a Python library for modeling resonant X-ray reflectivity (RXR) in oxide heterostructures. It provides:

- An object-oriented API for building and managing multilayer structures.  
- A Cython-accelerated backend for fast reflectivity calculations.  
- A local materials database of atomic form factors.  
- Utilities for plotting and visualizing reflectivity curves.  

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install rxrmask

Quick Start
-----------

Build a simple bilayer and compute its reflectivity:

.. code-block:: python

   import numpy as np
   from rxrmask.core.structure import Structure
   from rxrmask.core.layer import Layer
   from rxrmask.Pythonreflectivity import reflectivity

   # Define structure: Si (10 nm) on Cr (5 nm)
   s = Structure("Si/Cr", n_compounds=2)
   s.add_layer(0, Layer("Si", thickness=10.0, density=2.33))
   s.add_layer(1, Layer("Cr", thickness=5.0, density=7.19))

   # Compute reflectivity at 1000 eV over qz
   qz = np.linspace(0.0, 0.5, 1000)
   qz, R_sigma, R_pi = reflectivity(s, qz, E_eV=1000.0)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   api/backends
   api/core
   api/materials
   api/pint_init
   api/utils
   api/optimization
   examples

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
