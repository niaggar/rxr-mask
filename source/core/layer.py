from dataclasses import dataclass
import numpy as np
from .atom import Atom
from source.pint_init import ureg

r_e = 2.8179403262e-15 * ureg.m

@dataclass
class Layer:
    id: str
    thickness: float
    density: float
    atom: Atom

    def get_n(self, energy_eV: float):
        f1, f2 = self.atom.f1f2(energy_eV)

        h = 4.135667696e-15  # Plank's Constant [eV s]
        c = 2.99792450e10  # Speed of light in vacuum [cm/s]
        re = 2.817940322719e-13  # Classical electron radius (Thompson scattering length) [cm]
        avocado = 6.02214076e23  # avagoadro's number
        k0 = 2 * np.pi * energy_eV / (h * c)  # photon wavenumber in vacuum [1/cm]
        constant = 2 * np.pi * re * (avocado) / (k0 ** 2)  # constant for density sum

        delta =  constant * f1
        beta  =  constant * f2

        return 1 - delta + 1j*beta

    def get_thickness_angstrom(self):
        return self.thickness