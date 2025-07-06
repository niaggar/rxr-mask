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
    mass: float
    ff: FormFactorModel


def get_atom(symbol: str, atoms: list[Atom]) -> Atom | None:
    for atom in atoms:
        if atom.name == symbol:
            return atom
    return None
