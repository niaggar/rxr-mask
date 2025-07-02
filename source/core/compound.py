from dataclasses import dataclass, field
import numpy as np
from .atom import Atom, get_atom

from .layer import Layer


@dataclass
class Compound:
    id: str
    name: str
    thickness: float
    density: float
    atoms: list[Atom]
    layers: list[list[Layer]] = field(default_factory=list)
    n_layers: int = 1

    def create_layer(self, n_layers:int=1):
        self.n_layers = n_layers
        layers = []
        delta_thickness = self.thickness / n_layers

        for i in range(n_layers):
            levels = []
            thickness = delta_thickness

            for atom in self.atoms:
                layer = Layer(
                    id=f"{self.id}_{atom.symbol}_{i}",
                    thickness=thickness,
                    density=self.density,
                    atom=atom,
                )
                levels.append(layer)

            layers.append(levels)

        self.layers = layers
    
    def get_n_layer(self, energy_eV: float, layer_index: int):
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError("Layer index out of range.")

        n_complex = 0 + 0j
        for l in self.layers[layer_index]:
            n_complex += l.get_n(energy_eV) 

        return n_complex
    
    def get_thickness_layer(self, layer_index: int):
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError("Layer index out of range.")

        return self.layers[layer_index][0].get_thickness_angstrom()


def create_compound(id: str, name: str, thickness: float, density: float, formula: str, n_layers: int = 1) -> Compound:
    """
    Create a Compound instance from the given parameters.

    Parameters:
        id (str): Identifier for the compound.
        name (str): Name of the compound.
        thickness (float): Total thickness of the compound in angstroms.
        density (float): Density of the compound.
        formula (str): Chemical formula in the format 'Symbol:Count,Symbol:Count', e.g., 'C:2,O:1'.
        n_layers (int, optional): Number of layers to divide the compound into. Defaults to 1.

    Returns:
        Compound: The created Compound instance.
    """
    atoms = []
    for atom_info in formula.split(","):
        symbol, count = atom_info.strip().split(":")
        count = int(count)

        atom = get_atom(symbol)
        atoms.extend([atom] * count)

    compound = Compound(id=id, name=name, thickness=thickness, density=density, atoms=atoms)
    compound.create_layer(n_layers=n_layers)

    return compound
