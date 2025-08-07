import numpy as np


class ReflectivityData:
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
    energy_range: np.ndarray
    qz: float
    R_s: np.ndarray
    R_p: np.ndarray

    def __init__(self):
        self.energy_range = np.array([])
        self.qz = 0.0
        self.R_s = np.array([])
        self.R_p = np.array([])
