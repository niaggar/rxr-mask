"""Structure module for RXR-Mask.

This module provides the Structure class for representing complete multilayer
structures used in X-ray reflectometry calculations.
"""

from rxrmask.core import Compound, AtomLayer, Layer
from rxrmask.utils import get_density_profile_from_element_data

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
    
    def create_layers(self, step: float = 0.1, element_data=None, layer_thickness_params=None, atoms=None, params_container=None):
        from rxrmask.core.parameter import ParametersContainer
        
        # Get element data if not provided
        if element_data is None or layer_thickness_params is None or atoms is None:
            element_data, layer_thickness_params, atoms = self.get_element_data()
        
        # Create or use provided parameters container
        if params_container is None:
            params_container = ParametersContainer()
        
        # Calculate density profiles
        z, dens, m_dens, _ = get_density_profile_from_element_data(
            element_data, layer_thickness_params, atoms, step
        )
        
        # Create layers with managed parameters
        layers = []
        
        for i in range(1, len(z)):  # Skip first point (z=0)
            # Calculate layer thickness
            thickness = z[i] - z[i-1] if i > 0 else step
            
            # Create atomic elements for this layer
            elements = []
            for element_name, atom in atoms.items():
                molar_density = dens[element_name][i] if element_name in dens else 0.0
                molar_m_density = m_dens[element_name][i] if element_name in m_dens else 0.0
                
                # Create managed parameters for this layer's densities
                density_param = params_container.new_parameter(
                    f"layer_{i}_{element_name}_density", 
                    molar_density
                )
                mag_density_param = params_container.new_parameter(
                    f"layer_{i}_{element_name}_mag_density", 
                    molar_m_density
                )
                
                element_layer = AtomLayer(
                    atom=atom,
                    molar_density=density_param,
                    molar_magnetic_density=mag_density_param
                )
                elements.append(element_layer)
            
            # Create layer
            layer = Layer(
                id=f"Layer_{i}",
                thickness=thickness,
                elements=elements
            )
            layers.append(layer)
        
        self.layers = layers
        self.n_layers = len(layers)
        
        return layers, params_container
    
    def update_layers(self, element_data, layer_thickness_params, atoms, step: float = 0.1):
        """Update existing layers with new density profiles.
        
        This method efficiently updates the layer densities when parameters change,
        without recreating the entire layer structure. Useful during optimization
        when only density values change but the layer structure remains the same.
        
        Args:
            element_data (dict): Element data structure with updated parameter values.
            layer_thickness_params (list): List of thickness Parameter objects.
            atoms (dict): Dictionary mapping element names to Atom objects.
            step (float, optional): Discretization step size. Defaults to 0.1.
            
        Raises:
            ValueError: If no layers exist (must call create_layers first).
        """
        if not self.layers:
            raise ValueError("No layers exist. Call create_layers or create_layers_optimized first.")
        
        # Calculate updated density profiles
        z, dens, m_dens, _ = get_density_profile_from_element_data(
            element_data, layer_thickness_params, atoms, step
        )
        
        # Update existing layers with new density values
        for i, layer in enumerate(self.layers):
            layer_index = i + 1  # Layers start from index 1 (skip z=0)
            
            if layer_index < len(z):
                # Update thickness if needed
                new_thickness = z[layer_index] - z[layer_index-1] if layer_index > 0 else step
                layer.thickness = new_thickness
                
                # Update density values for each element in the layer
                for element_layer in layer.elements:
                    element_name = element_layer.atom.name
                    
                    if element_name in dens and layer_index < len(dens[element_name]):
                        # Update regular density
                        new_density = dens[element_name][layer_index]
                        element_layer.molar_density.set(new_density)
                        
                        # Update magnetic density
                        if element_name in m_dens and layer_index < len(m_dens[element_name]):
                            new_mag_density = m_dens[element_name][layer_index]
                            if element_layer.molar_magnetic_density is not None:
                                element_layer.molar_magnetic_density.set(new_mag_density)
    
    def get_layers_summary(self) -> dict:
        """Get a summary of the current layer structure.
        
        Returns:
            dict: Dictionary containing layer structure information including:
                - n_layers: Number of layers
                - total_thickness: Total structure thickness
                - elements: List of unique elements
                - layer_info: Detailed information for each layer
        """
        if not self.layers:
            return {
                'n_layers': 0,
                'total_thickness': 0.0,
                'elements': [],
                'layer_info': []
            }
        
        # Collect unique elements
        unique_elements = set()
        layer_info = []
        total_thickness = 0.0
        
        for i, layer in enumerate(self.layers):
            total_thickness += layer.thickness
            
            # Collect element information for this layer
            layer_elements = {}
            for element_layer in layer.elements:
                element_name = element_layer.atom.name
                unique_elements.add(element_name)
                
                layer_elements[element_name] = {
                    'molar_density': element_layer.molar_density.get(),
                    'molar_magnetic_density': (element_layer.molar_magnetic_density.get() 
                                             if element_layer.molar_magnetic_density is not None else 0.0)
                }
            
            layer_info.append({
                'id': layer.id,
                'thickness': layer.thickness,
                'elements': layer_elements
            })
        
        return {
            'n_layers': len(self.layers),
            'total_thickness': total_thickness,
            'elements': list(unique_elements),
            'layer_info': layer_info
        }




    def get_element_data(self):
        for compound in self.compounds:
            if compound is None:
                raise ValueError("All compounds must be defined to get element data.")
        
        valid_compounds = [c for c in self.compounds if c is not None]
        atoms = {}
        element_data = {}
        
        for i, compound in enumerate(valid_compounds):
            for element in compound.compound_details:
                element_name = element.name
                
                if element_name not in atoms:
                    atoms[element_name] = element.atom
                    element_data[element_name] = {
                        'density_params': [None] * (len(valid_compounds) + 1),
                        'magnetic_density_params': [None] * (len(valid_compounds) + 1),
                        'roughness_params': [None] * len(valid_compounds),
                        'thickness_params': [None] * len(valid_compounds)
                    }
                
                element_data[element_name]['density_params'][i] = element.molar_density
                element_data[element_name]['magnetic_density_params'][i] = element.molar_magnetic_density
                element_data[element_name]['roughness_params'][i] = element.roughness
                element_data[element_name]['thickness_params'][i] = element.thickness
        
        layer_thickness_params = [compound.thickness for compound in valid_compounds]
        return element_data, layer_thickness_params, atoms

    def get_density_profile(self, step: float = 0.1):
        element_data, layer_thickness_params, atoms = self.get_element_data()
        return get_density_profile_from_element_data(element_data, layer_thickness_params, atoms, step)
