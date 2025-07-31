"""Form factor handling for X-ray reflectometry calculations.

Provides interface and implementation for atomic form factor models.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

import pathlib
import rxrmask


@dataclass
class FormFactorModel:
    """Base class for atomic form factor models.
    
    Defines interface for X-ray form factor calculations.
    
    Attributes:
        ff_path (str | None): Path to form factor data file.
    """
    ff_path: str | None = field(default=None)

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        """Get form factors at specific energy.
        
        Args:
            energy_eV (float): X-ray energy in eV.
            
        Returns:
            tuple[float, float]: (f1, f2) form factor components.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_formfactors_energies(self, energies: np.ndarray) -> list[tuple[float, float]]:
        """Get form factors for multiple energies.
        
        Args:
            energies (np.ndarray): Array of energies in eV.
            
        Returns:
            list[tuple[float, float]]: List of (f1, f2) for each energy.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_all_formfactors(self, *args) -> pd.DataFrame:
        """Get complete form factor dataset.
        
        Returns:
            pd.DataFrame: All form factor data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_path(self) -> str | None:
        """Get path to form factor data file."""
        return self.ff_path


class FormFactorLocalDB(FormFactorModel):
    """Form factor model using local database files.
    
    Reads atomic form factors from local text files with support for
    magnetic and non-magnetic elements.
    
    Attributes:
        ff_data (pd.DataFrame | None): Loaded form factor data.
        element (str | None): Chemical element symbol.
        is_magnetic (bool): Whether to load magnetic form factors.
    """
    ff_data: pd.DataFrame | None = field(default=None)
    element: str | None = field(default=None)
    is_magnetic: bool = field(default=False)

    def __init__(self, element: str, is_magnetic: bool = False):
        """Initialize form factor model for specified element.
        
        Args:
            element (str): Chemical symbol (e.g., 'Fe', 'O').
            is_magnetic (bool): Load magnetic form factors.
        """
        super().__init__()
        self.element = element
        self.is_magnetic = is_magnetic
        self.read_data()

    def read_data(self):
        """Read form factor data from local database."""
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
            sep=r"\s+",
            header=None,
            index_col=False,
            names=["E", "f1", "f2"],
            dtype=float,
            comment="#",
        )

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        """Get form factors at specific energy using interpolation."""
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        f1 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f1))
        f2 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f2))
        
        return f1, f2
    
    def get_formfactors_energies(self, energies: np.ndarray) -> list[tuple[float, float]]:
        """Get form factors for multiple energies using interpolation."""
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        f1 = np.interp(energies, self.ff_data.E, self.ff_data.f1)
        f2 = np.interp(energies, self.ff_data.E, self.ff_data.f2)
        
        return list(zip(f1, f2))
    
    def get_all_formfactors(self, *args) -> pd.DataFrame:
        """Return complete form factor dataset."""
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        return self.ff_data
    