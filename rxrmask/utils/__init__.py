# Density profile utilities
from .density_profile import density_profile, z_density_profile, step_function

# Check if plot utilities are available and import if present
try:
    from .plot import plot_reflectivity, plot_energy_scan, plot_slab_model, plot_density_profile
    _plot_available = True
except ImportError:
    _plot_available = False

# Define what gets exported when using "from rxrmask.utils import *"
__all__ = [
    # Density profile functions
    'density_profile',
    'z_density_profile', 
    'step_function',
]

# Add plot utilities if available
if _plot_available:
    __all__.extend(['plot_reflectivity', 'plot_energy_scan', 'plot_slab_model', 'plot_density_profile'])