from rxrmask.core.compound import Compound
from rxrmask.core.layer import ElementLayer, Layer
from rxrmask.utils.density_profile import density_profile

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Structure:
    name: str
    n_compounds: int = 0
    n_layers: int = 0

    layers: list[Layer] = field(default_factory=list)
    compounds: list[Compound | None] = field(default_factory=list)

    def __init__(self, name: str, n_compounds: int):
        self.name = name
        self.n_compounds = n_compounds
        self.compounds = [None] * n_compounds

    def add_compound(self, index: int, compound: Compound):
        if index < 0 or index >= self.n_compounds:
            raise IndexError("Index out of range for compounds.")
        if not isinstance(compound, Compound):
            raise TypeError("Expected a Compound instance.")

        self.compounds[index] = compound

    def create_layers(self, step: float = 0.1):
        z, dens, m_dens, atoms = self.get_density_profile(step=step)
        layers = []

        for i, z_val in enumerate(z):
            if i == 0:
                continue

            elements = []
            for name, atom in atoms.items():
                molar_density = dens[name][i] if name in dens else 0.0
                molar_m_density = m_dens[name][i] if name in m_dens else 0.0

                element_layer = ElementLayer(
                    atom=atom,
                    molar_density=molar_density,
                    molar_magnetic_density=molar_m_density
                )
                elements.append(element_layer)

            layer = Layer(
                id=f"Layer_{i}",
                thickness=z_val - z[i-1],
                elements=elements
            )
            layers.append(layer)

        self.layers = layers
        self.n_layers = len(layers)

    def get_density_profile(self, step: float = 0.1):
        thicknesses = [c.thickness for c in self.compounds if c is not None]
        densities = {}
        m_densities = {}
        roughness = {}
        atoms = {}
        min_n_densities = len(self.compounds) + 1

        for i, c in enumerate(self.compounds):
            if c is None:
                raise ValueError("All compounds must be defined to get the density profile.")
            
            for element in c.formula_struct:
                if element.name not in densities:
                    densities[element.name] = np.zeros(min_n_densities).tolist()
                    m_densities[element.name] = np.zeros(min_n_densities).tolist()
                    roughness[element.name] = np.zeros(min_n_densities - 1).tolist()
                if element.name not in atoms:
                    atoms[element.name] = element.atom

                densities[element.name][i] = element.molar_density if element.molar_density is not None else 0
                m_densities[element.name][i] = element.molar_m_density if element.molar_m_density is not None else 0
                roughness[element.name][i] = element.roughness if element.roughness is not None else c.base_roughness

        z, dens = density_profile(thicknesses, densities, roughness, step=step)
        _, m_dens = density_profile(thicknesses, m_densities, roughness, step=step)
        return z, dens, m_dens, atoms




    def __repr__(self) -> str:
        return f"Structure(name={self.name}, n_compounds={self.n_compounds}, n_layers={self.n_layers})"

