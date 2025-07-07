from dataclasses import dataclass
import numpy as np
from .atom import Atom

@dataclass
class Layer:
    id: str
    thickness: float # in Angstroms
    density: float # in mol/cm^3
    atom: Atom

    def get_f1f2(self, energy_eV: float) -> tuple[float, float]:
        return self.atom.ff.get_f1f2(energy_eV)
    
    def set_density_gcm(self, density: float, total_mass: float):
        if self.atom.mass is None:
            raise ValueError("Atomic mass is not set for the atom.")

        self.density = density * self.atom.stochiometric_fraction / total_mass

    def get_thickness_angstrom(self):
        return self.thickness