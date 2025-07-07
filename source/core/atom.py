from source.core.formfactor import FormFactorModel
from source.pint_init import ureg, Q_

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd



@dataclass
class Atom:
    Z: int
    name: str
    symbol: str
    ff: FormFactorModel
    stochiometric_fraction: float = 1.0
    mass: float | None = None  # in atomic mass units (g/mol)

    def __post_init__(self):
        if not isinstance(self.ff, FormFactorModel):
            raise TypeError("ff must be an instance of FormFactorModel")
        
        if self.mass is None:
            self.mass = self.load_atomic_mass()
    
    def load_atomic_mass(self):
        mass = None
        
        # TODO: Use relative path to the atomic mass file
        file = open("/Users/niaggar/Developer/mitacs/rxr-mask/source/materials/atomic_mass.txt", "r")
        lines = file.readlines()
        for line in lines:
            if line.split()[0] == self.symbol:
                mass = line.split()[1]
        file.close()

        if mass == None:
            raise NameError("Inputted formula not found in perovskite density database")

        return float(mass)


def get_atom(symbol: str, atoms: list[Atom]) -> Atom | None:
    for atom in atoms:
        if atom.name == symbol:
            return atom
    return None
