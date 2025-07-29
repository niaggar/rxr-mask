"""Structure module for RXR-Mask.

This module provides the Structure class for representing complete multilayer
structures used in X-ray reflectometry calculations.
"""

from rxrmask.core.compound import Compound
from rxrmask.core.layer import ElementLayer, Layer
from rxrmask.utils.density_profile import density_profile

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Structure:
    """Represents a complete multilayer structure for X-ray reflectometry.
    
    This class manages the organization of compounds into a multilayer structure,
    handles the discretization into thin layers, and provides methods for
    generating density profiles. It serves as the main container that connects
    the material definitions (compounds) with the computational representation (layers).
    
    Attributes:
        name (str): Human-readable name for the structure.
        n_compounds (int): Number of compounds in the structure. Defaults to 0.
        n_layers (int): Number of discretized layers. Defaults to 0.
        layers (list[Layer]): List of discretized Layer objects.
        compounds (list[Compound | None]): List of Compound objects, may contain None entries.
    """
    name: str
    n_compounds: int = 0
    n_layers: int = 0

    layers: list[Layer] = field(default_factory=list)
    compounds: list[Compound | None] = field(default_factory=list)

    def __init__(self, name: str, n_compounds: int):
        """Initialize a new Structure with specified name and compound capacity.
        
        Args:
            name (str): Human-readable name for the structure.
            n_compounds (int): Number of compounds that will be added to the structure.
        """
        self.name = name
        self.n_compounds = n_compounds
        self.compounds = [None] * n_compounds

    def add_compound(self, index: int, compound: Compound):
        """Add a compound to the structure at the specified index.
        Index 0 represents the substrate layer, and subsequent indices
        represent the layers above it in the multilayer structure.
        
        Args:
            index (int): Position index where the compound should be added (0-based).
            compound (Compound): The Compound object to add to the structure.
            
        Raises:
            IndexError: If index is outside the valid range [0, n_compounds).
            TypeError: If compound is not a Compound instance.
        """
        if index < 0 or index >= self.n_compounds:
            raise IndexError("Index out of range for compounds.")
        if not isinstance(compound, Compound):
            raise TypeError("Expected a Compound instance.")

        self.compounds[index] = compound

    def create_layers(self, step: float = 0.1):
        """Create discretized layers from the compound definitions.
        
        Generates a series of thin layers by discretizing the compound structure
        based on the density profile. Each layer contains the appropriate
        atomic elements with their calculated densities at that position.
        
        This method transforms the compound-based description into a layer-based
        representation.
        
        Args:
            step (float, optional): Discretization step size in Angstroms. 
                                   Smaller values give higher resolution but more layers. 
                                   Defaults to 0.1.
        """
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
        """Generate density profiles for all atomic species in the structure.
        
        Calculates the spatial distribution of atomic densities throughout the
        structure, taking into account interface roughness and gradual transitions
        between compounds. This method provides the fundamental data needed for
        optical constant calculations.
        
        Args:
            step (float, optional): Spatial resolution for the density profile in Angstroms.
                                   Smaller values provide higher resolution. Defaults to 0.1.
                                   
        Returns:
            tuple: A 4-tuple containing:
                - z (np.ndarray): Spatial positions in Angstroms.
                - dens (dict): Regular density profiles for each atomic species.
                - m_dens (dict): Magnetic density profiles for each atomic species.  
                - atoms (dict): Dictionary mapping element names to Atom objects.
                
        Raises:
            ValueError: If any compound in the structure is None (undefined).
        """
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
        """Return a string representation of the Structure.
        
        Returns:
            str: A formatted string showing the structure's name, number of compounds,
                 and number of layers.
        """
        return f"Structure(name={self.name}, n_compounds={self.n_compounds}, n_layers={self.n_layers})"

