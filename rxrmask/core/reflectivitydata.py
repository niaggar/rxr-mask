import numpy as np


class ReflectivityData:
    """Class for storing reflectivity data.

    Attributes:
        qz: The momentum transfer vector.
        energy: The energy of the incident beam.
        R_s: The s-polarized reflectivity data.
        R_p: The p-polarized reflectivity data.
    """
    
    qz: np.ndarray
    energy: float
    R_s: np.ndarray
    R_p: np.ndarray
    
    def __init__(self):
        self.energy = 0.0
        self.qz = np.array([])
        self.R_s = np.array([])
        self.R_p = np.array([])

class EnergyScanData:
    """Class for storing energy scan data.

    Attributes:
        energy_range: The range of energies for the scan.
        qz: The momentum transfer vector.
        R_s: The s-polarized reflectivity data.
        R_p: The p-polarized reflectivity data.
    """
    energy_range: np.ndarray
    qz: float
    R_s: np.ndarray
    R_p: np.ndarray

    def __init__(self):
        self.energy_range = np.array([])
        self.qz = 0.0
        self.R_s = np.array([])
        self.R_p = np.array([])
