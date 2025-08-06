"""Computational backends for X-ray reflectometry calculations."""

from .pr_backend import (
    reflectivity as reflectivity_pr,
    reflectivity_parallel as reflectivity_parallel_pr,
    energy_scan as energy_scan_pr,
    energy_scan_parallel as energy_scan_parallel_pr,
)

try:
    from .udkm_backend import test as udkm_test
    _udkm_available = True
except ImportError:
    _udkm_available = False

__all__ = [
    'reflectivity_pr',
    'reflectivity_parallel_pr',
    'energy_scan_pr',
    'energy_scan_parallel_pr',
]

if _udkm_available:
    __all__.append('udkm_test')