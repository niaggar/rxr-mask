"""Atom module for RXR-Mask.

This module provides the Atom class and related utilities for representing
atomic species in X-ray reflectometry calculations.
"""

from rxrmask.core.formfactor import FormFactorModel

from dataclasses import dataclass
import pathlib
import rxrmask


@dataclass
class Atom:
    """Represents an atomic species for X-ray reflectometry calculations.
    
    This class stores atomic properties including atomic number, name, form factors,
    and atomic mass. The atomic mass is automatically loaded from a database if not provided.
    
    Attributes:
        Z (int): Atomic number.
        name (str): Chemical symbol of the atom (e.g., 'Fe', 'O').
        ff (FormFactorModel): X-ray form factor model for the atom.
        ffm (FormFactorModel | None): Magnetic form factor model for the atom. Defaults to None.
        mass (float | None): Atomic mass in atomic mass units (g/mol). 
                            Automatically loaded if not provided.
    
    Raises:
        TypeError: If ff is not an instance of FormFactorModel.
        NameError: If atomic mass cannot be found in the database.
    """
    Z: int
    name: str
    ff: FormFactorModel
    ffm: FormFactorModel | None = None
    mass: float = 0.0  # in atomic mass units (g/mol)

    def __post_init__(self):
        """Post-initialization to validate form factor and load atomic mass.
        
        Validates that the form factor is an instance of FormFactorModel and
        automatically loads the atomic mass if not provided.
        
        Raises:
            TypeError: If ff is not an instance of FormFactorModel.
        """
        if not isinstance(self.ff, FormFactorModel):
            raise TypeError("ff must be an instance of FormFactorModel")
        
        if self.mass == 0.0:
            self.mass = self.load_atomic_mass()
    
    def load_atomic_mass(self):
        """Load atomic mass from the materials database.
        
        Searches for the atomic mass in the atomic_mass.txt file based on the
        atom's name (chemical symbol).
        
        Returns:
            float: Atomic mass in atomic mass units (g/mol).
            
        Raises:
            NameError: If the atom's name is not found in the database.
        """
        mass = None
        data_path = pathlib.Path(rxrmask.__file__).parent / "materials" / "atomic_mass.txt"
        
        with open(data_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.split()[0] == self.name:
                    mass = line.split()[1]
            file.close()

        if mass == None:
            raise NameError("Inputted formula not found in perovskite density database")

        return float(mass)


def find_atom(symbol: str, atoms: list[Atom]) -> Atom | None:
    """Find an atom by its chemical symbol from a list of atoms.
    
    Searches through a list of Atom objects to find one with the specified
    chemical symbol.
    
    Args:
        symbol (str): Chemical symbol to search for (e.g., 'Fe', 'O').
        atoms (list[Atom]): List of Atom objects to search through.
        
    Returns:
        Atom | None: The first Atom object with matching symbol, or None if not found.
    """
    for atom in atoms:
        if atom.name == symbol:
            return atom
    return None
