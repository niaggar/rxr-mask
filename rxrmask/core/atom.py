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
    """Atomic species for X-ray reflectometry calculations.

    Stores atomic properties, form factors, and mass. Mass is auto-loaded from database.

    Attributes:
        Z: Atomic number
        name: Chemical symbol (e.g., 'Fe', 'O')
        ff: X-ray form factor model
        ffm: Magnetic form factor model (optional)
        mass: Atomic mass in g/mol (auto-loaded if 0.0)
    """

    Z: int
    name: str
    ff: FormFactorModel
    ffm: FormFactorModel | None = None
    mass: float = 0.0  # in atomic mass units (g/mol)

    def __post_init__(self):
        """Validate form factor and auto-load atomic mass."""
        if not isinstance(self.ff, FormFactorModel):
            raise TypeError("ff must be an instance of FormFactorModel")

        if self.mass == 0.0:
            self.mass = self.load_atomic_mass()

    def load_atomic_mass(self):
        """Load atomic mass from materials database.

        Returns:
            float: Atomic mass in g/mol

        Raises:
            NameError: If atom symbol not found in database
        """
        # Validate if the name of the atom is somthing like Xi where i is a number
        if len(self.name) >= 2:
            if (
                self.name[0].isalpha() and self.name[0].upper() == "X"
                and self.name[1:].isdigit()
            ):
                print(
                    f"Warning: Atom name '{self.name}' is a placeholder. Atomic mass set to 0.0."
                )
                return 0.0

        mass = None
        data_path = (
            pathlib.Path(rxrmask.__file__).parent / "materials" / "atomic_mass.txt"
        )

        with open(data_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.split()[0] == self.name:
                    mass = line.split()[1]
            file.close()

        if mass == None:
            raise NameError("Atom symbol not found in database")

        return float(mass)


def find_atom(symbol: str, atoms: list[Atom]) -> Atom | None:
    """Find atom by chemical symbol.

    Args:
        symbol: Chemical symbol (e.g., 'Fe', 'O')
        atoms: List of Atom objects to search

    Returns:
        Atom object if found, None otherwise
    """
    for atom in atoms:
        if atom.name == symbol:
            return atom
    return None
