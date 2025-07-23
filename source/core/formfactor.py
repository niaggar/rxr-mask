from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class FormFactorModel:
    ff_path: str | None = field(default=None)

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_all_formfactors(self, *args) -> pd.DataFrame:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_path(self) -> str | None:
        return self.ff_path
