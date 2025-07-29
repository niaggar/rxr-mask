# Version information
__version__ = "0.1.0"

# Make core modules easily accessible
from . import core
from . import backends
from . import utils

# Optionally expose the most commonly used classes at the top level
from .core import Atom, Structure, Compound, Layer, FormFactorLocalDB

__all__ = [
    # Version info
    '__version__',
    
    # Submodules
    'core',
    'backends', 
    'utils',
    
    # Commonly used classes
    'Atom',
    'Structure',
    'Compound',
    'Layer',
    'FormFactorLocalDB',
]