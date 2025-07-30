"""Compound module for RXR-Mask.

This module provides classes and utilities for representing chemical compounds
and their atomic layer structures in X-ray reflectometry calculations.
"""

from rxrmask.core.atom import Atom, find_atom
from rxrmask.core.parameter import Parameter, ParametersContainer

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CompoundDetails:
    """Represents an atomic species within a compound layer structure.

    This class stores information about a specific atomic species in a compound,
    including its stoichiometric fraction, associated atom object, and layer-specific
    properties.

    Attributes:
        name (str): Chemical symbol of the atom (e.g., 'Fe', 'O').
        stochiometric_fraction (float): Stoichiometric coefficient in the compound formula.
        atom (Atom): The Atom object containing atomic properties.
        thickness (Parameter[float]): Thickness of the layer in Angstroms.
        molar_density (Parameter[float]): Molar density in mol/cm³.
        roughness (Parameter[float]): Interface roughness in Angstroms. Defaults to 0.0.
        molar_magnetic_density (Parameter[float]): Molar magnetic density in mol/cm³. Defaults to 0.0.
    """

    name: str
    stochiometric_fraction: float
    atom: Atom
    thickness: Parameter[float]  # in Angstrom
    roughness: Parameter[float] # in Angstrom
    molar_density: Parameter[float]  # in mol/cm^3
    molar_magnetic_density: Parameter[float] # in mol/cm^3

    def __init__(self):
        """Initialize a new CompoundDetails instance with default values."""
        self.name = ""
        self.stochiometric_fraction = 0.0


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

    name: str
    formula: str

    thickness: Parameter[float]  # in Angstrom
    roughness: Parameter[float] # in Angstrom
    density: Parameter[float]  # in g/cm^3
    magnetic_density: Parameter[float] # in g/cm^3

    magnetic: bool = False
    magnetic_direction: Literal["x", "y", "z", "0"] = "0"
    compound_details: list[CompoundDetails] = field(default_factory=list)

    def __init__(self):
        self.name = ""
        self.formula = ""
        self.compound_details = []
        self.magnetic = False
        self.magnetic_direction = "0"


def get_compound_details(formula: str, atoms: list[Atom]) -> list[CompoundDetails]:
    """Extract detailed information about the compounds from the given formula.

    Args:
        formula (str): The chemical formula to parse.
        atoms (list[Atom]): List of available Atom objects.

    Returns:
        list[CompoundDetails]: A list of CompoundDetails objects containing
                                information about each element in the formula.
    """
    details = []
    for part in formula.split(","):
        element, count = part.split(":")
        atom = find_atom(element, atoms)
        if atom is None:
            raise ValueError(f"Unknown atom: {element}")

        detail = CompoundDetails()
        detail.name = element
        detail.stochiometric_fraction = int(count)
        detail.atom = atom
        details.append(detail)

    return details


def create_compound(
    parameters_container: ParametersContainer,
    name: str,
    formula: str,
    thickness: float,
    density: float,
    atoms: list[Atom],
    roughness: float = 0.0,
    magetic_density: float = 0.0,
) -> Compound:
    compound_details = get_compound_details(formula, atoms)

    total_mass = 0.0
    for detail in compound_details:
        total_mass += detail.atom.mass * detail.stochiometric_fraction
        
    for detail in compound_details:
        frac = detail.stochiometric_fraction / total_mass
        base_name = f"{name}-{detail.name}"
        detail.molar_density = parameters_container.new_parameter(f"{base_name}-molar_density", density * frac)
        detail.molar_magnetic_density = parameters_container.new_parameter(f"{base_name}-molar_magnetic_density", magetic_density * frac)
        detail.roughness = parameters_container.new_parameter(f"{base_name}-roughness", roughness)
        detail.thickness = parameters_container.new_parameter(f"{base_name}-thickness", thickness)

    compound = Compound()
    compound.name = name
    compound.formula = formula
    compound.thickness = parameters_container.new_parameter(f"{name}-thickness", thickness)
    compound.density = parameters_container.new_parameter(f"{name}-density", density)
    compound.magnetic_density = parameters_container.new_parameter(f"{name}-magnetic_density", magetic_density)
    compound.roughness = parameters_container.new_parameter(f"{name}-roughness", roughness)
    compound.compound_details = compound_details

    return compound
