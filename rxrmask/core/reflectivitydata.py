import numpy as np
from typing import List, Tuple, Literal
from dataclasses import dataclass


class SimReflectivityData:
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


class SimEnergyScanData:
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


@dataclass
class ReflectivityScan:
    name: str
    energy_eV: float
    pol: Literal["s", "p"]
    qz: np.ndarray
    R: np.ndarray
    bounds: List[Tuple[float, float]]
    weights: List[float]
    background_shift: float = 0.0
    scale_factor: float = 1.0


@dataclass
class EnergyScan:
    name: str
    theta_deg: float
    pol: Literal["s", "p"]
    E_eV: np.ndarray
    R: np.ndarray
    bounds: List[Tuple[float, float]]
    weights: List[float]
    background_shift: float = 0.0
    scale_factor: float = 1.0
