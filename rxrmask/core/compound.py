from rxrmask.core.atom import Atom, get_atom

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AtomLayerStructure:
    name: str
    stochiometric_fraction: float
    atom: Atom

    molar_density: float | None = None  # in g/cm^3
    molar_m_density: float | None = None  # in mol/cm^3
    roughness: float | None = None  # in Angstrom
    thickness: float | None = None  # in Angstrom


@dataclass
class Compound:
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
