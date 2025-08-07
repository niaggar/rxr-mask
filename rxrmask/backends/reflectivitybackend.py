import numpy as np
from rxrmask.core import Structure
from rxrmask.core.reflectivitydata import ReflectivityData, EnergyScanData


class ReflectivityBackend:
    def compute_reflectivity(self, structure, qz, energy: float) -> ReflectivityData:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute_energy_scan(self, structure, energy_range, theta: float) -> EnergyScanData:
        raise NotImplementedError("This method should be implemented by subclasses.")
    