"""Form factor handling for X-ray reflectometry calculations.

Provides interface and implementation for atomic form factor models.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

import pathlib
import rxrmask


@dataclass
class FormFactorModel:
    """Base class for atomic form factor models.
    
    Defines interface for X-ray form factor calculations.
    
    Attributes:
        ff_path (str | None): Path to form factor data file.
    """
    ff_path: Optional[str] = field(default=None)

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
    
    def get_all_formfactors(self, *args) -> np.ndarray:
        """Get complete form factor dataset.
        
        Returns:
            np.ndarray: All form factor data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_path(self) -> str | None:
        """Get path to form factor data file."""
        return self.ff_path

@dataclass
class FormFactorLocalDB(FormFactorModel):
    """Form factor model using local database files with numpy."""
    element: Optional[str] = field(default=None)
    is_magnetic: bool = field(default=False)
    ff_data: Optional[np.ndarray] = field(default=None)

    def __init__(self, element: str, is_magnetic: bool = False):
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

        self.ff_data = np.loadtxt(
            file,
            comments="#",
            dtype=np.float64
        )

        if self.ff_data.shape[1] != 3:
            raise ValueError(f"Form factor file for '{self.element}' must have exactly 3 columns: E, f1, f2")

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")

        energies = self.ff_data[:, 0]
        f1 = np.interp(energy_eV, energies, self.ff_data[:, 1])
        f2 = np.interp(energy_eV, energies, self.ff_data[:, 2])
        return float(f1), float(f2)

    def get_formfactors_energies(self, energies: np.ndarray) -> list[tuple[float, float]]:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")

        f1 = np.interp(energies, self.ff_data[:, 0], self.ff_data[:, 1])
        f2 = np.interp(energies, self.ff_data[:, 0], self.ff_data[:, 2])
        return list(zip(f1, f2))

    def get_all_formfactors(self, *args) -> np.ndarray:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        return self.ff_data
