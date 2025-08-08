from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from typing import Optional

import pathlib
import rxrmask


@dataclass
class FormFactorModel:
    """Base class for atomic form factor models.
    
    Attributes:
        ff_path: Path to form factor data file
    """
    ff_path: Optional[str] = field(default=None)

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        """Get form factors at specific energy.
        
        Args:
            energy_eV: X-ray energy in eV
            args: Additional arguments for specific models
            
        Returns:
            (f1, f2) form factor components
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_formfactors_energies(self, energies: npt.NDArray[np.float64], *args) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get form factors for multiple energies.
        
        Args:
            energies: Array of energies in eV
            args: Additional arguments for specific models
            
        Returns:
            (f1, f2) form factor arrays
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_all_formfactors(self, *args) -> npt.NDArray[np.float64]:
        """Get complete form factor dataset.
        
        Returns:
            All form factor data
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_path(self) -> str | None:
        """Get path to form factor data file."""
        return self.ff_path
    
@dataclass
class FormFactorVacancy(FormFactorModel):
    """Form factor model representing a vacancy (zero form factor)."""
    
    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        return 0.0, 0.0
    
    def get_formfactors_energies(self, energies: npt.NDArray[np.float64], *args) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return np.zeros(len(energies)), np.zeros(len(energies))
    
    def get_all_formfactors(self, *args) -> npt.NDArray[np.float64]:
        return np.zeros((1, 3))

@dataclass
class FormFactorLocalDB(FormFactorModel):
    """Form factor model using local database files."""
    element: Optional[str] = field(default=None)
    is_magnetic: bool = field(default=False)
    ff_data: Optional[npt.NDArray[np.float64]] = field(default=None)

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

    def get_formfactors_energies(self, energies: npt.NDArray[np.float64], *args) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")

        f1 = np.interp(energies, self.ff_data[:, 0], self.ff_data[:, 1])
        f2 = np.interp(energies, self.ff_data[:, 0], self.ff_data[:, 2])
        return f1, f2

    def get_all_formfactors(self, *args) -> npt.NDArray[np.float64]:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        return self.ff_data
