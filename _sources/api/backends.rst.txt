Backends Package
================

The **backends** package provides a uniform interface for computing resonant X-ray reflectivity (RXR) using different computational engines. This module decouples the physical model definitions from their numerical implementations. This design makes it easy to maintain a single set of material and structural definitions that can be “translated” into multiple backend implementations for RXR calculations.

Currently, two backend implementations are available:

- **Pythonreflectivity**.
- **Udkm1Dsim**.

Pythonreflectivity Backend
---------------------------

**Main backend**.

.. automodule:: rxrmask.backends.pr_backend
   :members:
   :undoc-members:
   :show-inheritance:

Udkm1Dsim Backend
-----------------

**Work in progress**.

.. automodule:: rxrmask.backends.udkm_backend
   :members:
   :undoc-members:
   :show-inheritance:
