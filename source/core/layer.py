from dataclasses import dataclass, field
import numpy as np
from .atom import Atom


h = 4.135667696e-15  # Plank's Constant [eV s]
c = 2.99792450e10  # Speed of light in vacuum [cm/s]
re = 2.817940322719e-13  # Classical electron radius (Thompson scattering length) [cm]
avocado = 6.02214076e23  # avagoadro's number
re = 2.817940322719e-13 # Classical electron radius (Thompson scattering length) [cm]
pihc = 2 * np.pi / (h * c)  # To calculate the photon wavenumber in vacuum [1/cm]
pireav = 2 * np.pi * re * avocado  # To calculate the electron density in cm^-3


@dataclass
class ElementLayer:
    atom: Atom
    molar_density: float # in mol/cm^3
    molar_magnetic_density: float | None = None  # in mol/cm^3

    def get_f1f2(self, energy_eV: float) -> tuple[float, float]:
        return self.atom.ff.get_formfactors(energy_eV)
    
    def get_q1q2(self, energy_eV: float) -> tuple[float, float]:
        if self.atom.ffm is None:
            raise ValueError("The atom's magnetic form factor model is not set.")
        return self.atom.ffm.get_formfactors(energy_eV)


@dataclass
class Layer:
    id: str
    thickness: float # in Angstroms
    elements: list[ElementLayer] = field(default_factory=list)

    def get_index_of_refraction(self, energy_eV: float):
        f1_layer = 0.0
        f2_layer = 0.0
        for l in self.elements:
            f1, f2 = l.get_f1f2(energy_eV)
            f1_layer += f1 * l.molar_density
            f2_layer += f2 * l.molar_density

        k0 = energy_eV * pihc
        constant = pireav / (k0 ** 2)
        
        delta = constant * f1_layer
        beta  = constant * f2_layer
        n_complex = 1 - delta + 1j * beta

        return n_complex

    def get_magnetic_optical_constant(self, energy_eV: float):
        q1_layer = 0.0
        q2_layer = 0.0
        for l in self.elements:
            q1, q2 = l.get_q1q2(energy_eV)
            q1_layer += q1 * l.molar_density
            q2_layer += q2 * l.molar_density

        k0 = energy_eV * pihc
        constant = pireav / (k0 ** 2)

        delta_m = constant * q1_layer
        beta_m  = constant * q2_layer

        Q = 2 * (delta_m + 1j * beta_m)
        return Q