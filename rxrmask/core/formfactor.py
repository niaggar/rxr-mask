from dataclasses import dataclass, field
import pandas as pd
import numpy as np

import pathlib
import rxrmask


@dataclass
class FormFactorModel:
    ff_path: str | None = field(default=None)

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_all_formfactors(self, *args) -> pd.DataFrame:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_path(self) -> str | None:
        return self.ff_path


class FormFactorLocalDB(FormFactorModel):
    ff_data: pd.DataFrame | None = field(default=None)
    element: str | None = field(default=None)
    is_magnetic: bool = field(default=False)

    def __init__(self, element: str, is_magnetic: bool = False):
        super().__init__()
        self.element = element
        self.is_magnetic = is_magnetic
        self.read_data()

    def read_data(self):
        if self.element is None:
            raise ValueError("Element must be specified to read form factor data.")
        
        folder = "magnetic_form_factor" if self.is_magnetic else "form_factor"
        data_path = pathlib.Path(rxrmask.__file__).parent / "materials" / folder / f"{self.element}.txt"        
        self.ff_path = data_path.resolve().as_posix()
                
        file = pathlib.Path(self.ff_path)
        if not file.exists():
            raise FileNotFoundError(f"Form factor data file '{self.ff_path}' does not exist.")
        
        self.ff_data = pd.read_csv(
            file,
            sep="\s+",
            header=None,
            index_col=False,
            names=["E", "f1", "f2"],
            dtype=float,
            comment="#",
        )

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        f1 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f1))
        f2 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f2))
        
        return f1, f2
    
    def get_all_formfactors(self, *args) -> pd.DataFrame:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        return self.ff_data
    