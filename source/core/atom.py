from source.pint_init import ureg, Q_

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

FORM_FACTOR_DIR = Path(__file__).parents[1] / "materials" / "form_factor"

@dataclass
class Atom:
    Z: int
    symbol: str
    mass: float
    _ff: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def f1f2(self, energy_eV: float, *args) -> tuple[float, float]:
        self._ensure_loaded()
        df = self._ff
        
        if df is None:
            raise ValueError(f"Form factor data for '{self.symbol}' could not be loaded.")
        
        f1 = float(np.interp(energy_eV, df.E, df.f1))
        f2 = float(np.interp(energy_eV, df.E, df.f2))
        
        return f1, f2

    def _ensure_loaded(self):
        if self._ff is None:
            file = FORM_FACTOR_DIR / f"{self.symbol}.txt"

            if not file.exists():
                raise FileNotFoundError(f"No hay tabla f1/f2 para '{self.symbol}' en {file}")

            self._ff = pd.read_csv(
                file,
                delimiter="\t",
                header=None,
                index_col=False,
                names=["E", "f1", "f2"],
                dtype=float,
                comment="#",
            )


def get_atom(symbol: str) -> Atom:
    """
    Returns an Atom instance for the given symbol.
    NOTE: This is a stub implementation that always returns Z=1 and mass=1.
    Replace with a lookup for real atomic numbers and masses as needed.
    """
    symbol = symbol.capitalize()
    file = FORM_FACTOR_DIR / f"{symbol}.txt"

    if not file.exists():
        raise FileNotFoundError(f"No hay tabla f1/f2 para '{symbol}' en {file}")

    Z = 1  # TODO: Replace with actual atomic number lookup
    mass = 1  # TODO: Replace with actual atomic mass lookup
    return Atom(Z=Z, symbol=symbol, mass=mass)
