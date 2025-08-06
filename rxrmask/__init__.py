"""RXR-Mask: X-ray reflectometry analysis package.

Provides tools for X-ray reflectometry calculations and analysis.
"""

__version__ = "0.2.0"

from . import core
from . import backends
from . import utils
from .core import Atom, Structure, Compound, Layer, FormFactorLocalDB

__all__ = [
    '__version__',
    'core',
    'backends', 
    'utils',
    'Atom',
    'Structure',
    'Compound',
    'Layer',
    'FormFactorLocalDB',
]