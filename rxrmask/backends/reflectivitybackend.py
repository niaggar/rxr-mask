import numpy as np
from rxrmask.core import Structure
from rxrmask.core.reflectivitydata import ReflectivityData, EnergyScanData


class ReflectivityBackend:
    """Base class for reflectivity backends."""

    def compute_reflectivity(self, structure, qz, energy: float) -> ReflectivityData:
        """Compute the reflectivity for a given structure, qz values, and energy."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute_energy_scan(self, structure, energy_range, theta: float) -> EnergyScanData:
        """Compute the energy scan for a given structure, energy range, and theta."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    