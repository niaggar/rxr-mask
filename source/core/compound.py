from dataclasses import dataclass, field
import numpy as np
from .atom import Atom, get_atom

from .layer import Layer


h = 4.135667696e-15  # Plank's Constant [eV s]
c = 2.99792450e10  # Speed of light in vacuum [cm/s]
re = 2.817940322719e-13  # Classical electron radius (Thompson scattering length) [cm]
avocado = 6.02214076e23  # avagoadro's number
re = 2.817940322719e-13 # Classical electron radius (Thompson scattering length) [cm]
pihc = 2 * np.pi / (h * c)  # To calculate the photon wavenumber in vacuum [1/cm]
pireav = 2 * np.pi * re * avocado  # To calculate the electron density in cm^-3


@dataclass
class Compound:
    id: str
    name: str
    formula: str
    thickness: float
    # TODO: density should depend on the atoms and their counts 
    density: float # in g/cm^3
    atoms: list[Atom]
    layers: list[list[Layer]] = field(default_factory=list)
    n_layers: int = 1

    def create_layer(self, n_layers:int=1):
        self.n_layers = n_layers
        layers = []
        delta_thickness = self.thickness / n_layers

        total_mass = sum(atom.mass * atom.stochiometric_fraction for atom in self.atoms) # type: ignore

        for i in range(n_layers):
            levels = []
            thickness = delta_thickness

            for atom in self.atoms:
                layer = Layer(
                    id=f"{self.id}_{atom.name}_{i}",
                    thickness=thickness,
                    density=self.density,
                    atom=atom,
                )
                layer.set_density_gcm(self.density, total_mass)
                levels.append(layer)

            layers.append(levels)

        self.layers = layers
    
    def get_n_layer(self, energy_eV: float, layer_index: int):
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError("Layer index out of range.")

        f1_layer = 0.0
        f2_layer = 0.0
        for l in self.layers[layer_index]:
            f1, f2 = l.get_f1f2(energy_eV)
            f1_layer += f1 * l.density
            f2_layer += f2 * l.density


        k0 = energy_eV * pihc
        constant = pireav / (k0 ** 2)  # constant for density sum
        
        delta = constant * f1_layer
        beta  = constant * f2_layer
        n_complex = 1 - delta + 1j * beta

        return n_complex
    
    def get_thickness_layer(self, layer_index: int):
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError("Layer index out of range.")

        return self.layers[layer_index][0].get_thickness_angstrom()


def create_compound(id: str, name: str, thickness: float, density: float, formula: str, atoms_prov: list[Atom], n_layers: int = 1) -> Compound:
    atoms = []
    for atom_info in formula.split(","):
        symbol, count = atom_info.strip().split(":")
        count = int(count)
        
        atom = get_atom(symbol, atoms_prov)
        if atom is None:
            raise ValueError(f"Atom with symbol '{symbol}' not found in the provided atoms list.")
        
        atoms.extend([atom] * count)

    compound = Compound(id=id, name=name, thickness=thickness, density=density, atoms=atoms, formula=formula)
    
    # TODO: The layers shold be created using adaptative layer segmentation
    compound.create_layer(n_layers=n_layers)

    return compound
