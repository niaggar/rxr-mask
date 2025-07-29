from .pr_backend import (
    reflectivity,
    reflectivity_parallel,
    energy_scan,
    energy_scan_parallel,
    # Physical constants
    H_CONST,
    C_CONST,
    QZ_SCALE,
    HC_EV_ANGSTROM,
)

try:
    from .udkm_backend import test as udkm_test
    _udkm_available = True
except ImportError:
    _udkm_available = False

__all__ = [
    # Pythonreflectivity backend functions
    'reflectivity',
    'reflectivity_parallel', 
    'energy_scan',
    'energy_scan_parallel',
    
    # Physical constants
    'H_CONST',
    'C_CONST', 
    'QZ_SCALE',
    'HC_EV_ANGSTROM',
]

if _udkm_available:
    __all__.append('udkm_test')