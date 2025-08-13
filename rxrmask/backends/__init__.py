"""Computational backends for X-ray reflectometry calculations."""

from .backend import ReflectivityBackend
from .pr_backend import (
    PRReflectivityBackend,
    PRParallelReflectivityBackend,
)

try:
    from .udkm_backend import test as udkm_test

    _udkm_available = True
except ImportError:
    _udkm_available = False

__all__ = [
    "PRReflectivityBackend",
    "ReflectivityBackend",
    "PRParallelReflectivityBackend",
]

if _udkm_available:
    __all__.append("udkm_test")
