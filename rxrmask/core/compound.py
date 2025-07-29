"""Compound module for RXR-Mask.

This module provides classes and utilities for representing chemical compounds
and their atomic layer structures in X-ray reflectometry calculations.
"""

from rxrmask.core.atom import Atom, get_atom

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AtomLayerStructure:
    """Represents an atomic species within a compound layer structure.
    
    This class stores information about a specific atomic species in a compound,
    including its stoichiometric fraction, associated atom object, and layer-specific
    properties.
    
    Attributes:
        name (str): Chemical symbol of the atom (e.g., 'Fe', 'O').
        stochiometric_fraction (float): Stoichiometric coefficient in the compound formula.
        atom (Atom): The Atom object containing atomic properties.
        molar_density (float | None): Molar density in g/cm³. Defaults to None.
        molar_m_density (float | None): Magnetic molar density in mol/cm³. Defaults to None.
        roughness (float | None): Interface roughness in Angstroms. Defaults to None.
        thickness (float | None): Layer thickness in Angstroms. Defaults to None.
    """
    name: str
    stochiometric_fraction: float
    atom: Atom

    molar_density: float | None = None  # in g/cm^3
    molar_m_density: float | None = None  # in mol/cm^3
    roughness: float | None = None  # in Angstrom
    thickness: float | None = None  # in Angstrom


@dataclass
class Compound:
    """Represents a chemical compound in a multilayer structure.
    
    This class encapsulates all properties of a chemical compound including its
    composition, physical properties, and structural parameters. It can represent
    both magnetic and non-magnetic.
    
    Attributes:
        id (str): Unique identifier for the compound.
        name (str): Human-readable name of the compound.
        formula (str): Chemical formula in the format "Element1:count1,Element2:count2".
        thickness (float): Total thickness of the compound in Angstroms.
        base_density (float): Base density in g/cm³.
        base_m_density (float): Base molar density in mol/cm³. Defaults to 0.0.
        base_roughness (float): Base interface roughness in Angstroms. Defaults to 0.0.
        n_layers (int): Number of layers in the compound. Defaults to 1.
        id_layer_init (int): Initial layer ID. Defaults to 0.
        id_layer_end (int): Final layer ID. Defaults to 0.
        formula_struct (list[AtomLayerStructure]): List of atomic layer structures.
        magnetic (bool): Whether the compound is magnetic. Defaults to False.
        magnetic_direction (Literal["x", "y", "z", "0"]): Magnetic moment direction. 
                                                         Defaults to "0" (non-magnetic).
    """
    id: str
    name: str
    formula: str
    thickness: float
    
    base_density: float             # in g/cm^3
    base_m_density: float = 0.0     # in mol/cm^3
    base_roughness: float = 0.0     # in Angstrom

    n_layers: int = 1
    id_layer_init: int = 0
    id_layer_end: int = 0
    
    formula_struct: list[AtomLayerStructure] = field(default_factory=list)

    magnetic: bool = False
    magnetic_direction: Literal["x", "y", "z", "0"] = "0"



def create_compound(id: str, name: str, thickness: float, density: float, formula: str, atoms_prov: list[Atom], roughness: float | None = None, m_density: float | None = None) -> Compound:
    """Create a Compound object from a chemical formula and properties.
    
    Parses a chemical formula string and creates a complete Compound object with
    calculated atomic layer structures. The function automatically calculates
    molar densities for each atomic species based on their stoichiometric fractions
    and atomic masses.
    
    Args:
        id (str): Unique identifier for the compound.
        name (str): Human-readable name of the compound.
        thickness (float): Total thickness of the compound in Angstroms.
        density (float): Base density of the compound in g/cm³.
        formula (str): Chemical formula in the format "Element1:count1,Element2:count2"
                      (e.g., "Fe:1,O:3" for Fe₂O₃).
        atoms_prov (list[Atom]): List of available Atom objects to use in the compound.
        roughness (float | None, optional): Interface roughness in Angstroms. Defaults to None.
        m_density (float | None, optional): Magnetic density in mol/cm³. Defaults to None.
        
    Returns:
        Compound: A complete Compound object with calculated atomic layer structures.
        
    Raises:
        ValueError: If an atom symbol from the formula is not found in atoms_prov.
        
    Example:
        >>> atoms = [fe_atom, o_atom]  # Pre-defined Atom objects
        >>> compound = create_compound("fe2o3", "Iron Oxide", 50.0, 5.24, "Fe:2,O:3", atoms)
    """
    atoms = []
    formula_dict = []

    for atom_info in formula.split(","):
        symbol, count = atom_info.strip().split(":")
        count = int(count)
        
        atom = get_atom(symbol, atoms_prov)
        if atom is None:
            raise ValueError(f"Atom with symbol '{symbol}' not found in the provided atoms list.")
        
        found = False
        for elem in formula_dict:
            if elem.name == symbol:
                elem.stochiometric_fraction += count
                found = True
                break
        if not found:
            formula_dict.append(AtomLayerStructure(name=symbol, stochiometric_fraction=count, atom=atom))
        
        atoms.extend([atom])
    
    total_mass = 0
    for elem in formula_dict:
        atom = get_atom(elem.name, atoms)
        if atom is None:
            raise ValueError(f"Atom with name '{elem.name}' not found in the provided atoms list.")
        total_mass += atom.mass * elem.stochiometric_fraction

    for elem in formula_dict:
        elem.molar_density = density * elem.stochiometric_fraction / total_mass
        elem.molar_m_density = m_density * elem.stochiometric_fraction / total_mass if m_density is not None else None
        elem.roughness = roughness

    compound = Compound(id=id, name=name, thickness=thickness, base_density=density, formula=formula, formula_struct=formula_dict)
    return compound
