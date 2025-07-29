"""Form factor module for RXR-Mask.

This module provides a general class interface for handling X-ray atomic form factors.
All form factor models must implement this interface to ensure consistent behavior.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

import pathlib
import rxrmask


@dataclass
class FormFactorModel:
    """Abstract base class for atomic form factor models.
    
    This class defines the interface for form factor models used in X-ray
    reflectometry calculations. Subclasses must implement all abstract methods.
    
    Attributes:
        ff_path (str | None): Path to the form factor data file. If exists.
    """
    ff_path: str | None = field(default=None)

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        """Get atomic form factors at a specific energy.
        
        Args:
            energy_eV (float): X-ray energy in electron volts.
            *args: Additional arguments for specific implementations.
            
        Returns:
            tuple[float, float]: Tuple containing (f1, f2) form factor components.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_all_formfactors(self, *args) -> pd.DataFrame:
        """Get the complete form factor dataset.
        
        Args:
            *args: Additional arguments for specific implementations.
            
        Returns:
            pd.DataFrame: DataFrame containing all form factor data.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_path(self) -> str | None:
        """Get the path to the form factor data file.
        
        Returns:
            str | None: Path to the form factor data file, or None if not set.
        """
        return self.ff_path


class FormFactorLocalDB(FormFactorModel):
    """Form factor model using local database files.
    
    This class implements the FormFactorModel interface for reading atomic form
    factors from local text files. It supports both regular and magnetic form
    factors, automatically loading the appropriate data based on the element
    and magnetic properties.
    
    Attributes:
        ff_data (pd.DataFrame | None): Loaded form factor data. Defaults to None.
        element (str | None): Chemical symbol of the element. Defaults to None.
        is_magnetic (bool): Whether to load magnetic form factors. Defaults to False.
    """
    ff_data: pd.DataFrame | None = field(default=None)
    element: str | None = field(default=None)
    is_magnetic: bool = field(default=False)

    def __init__(self, element: str, is_magnetic: bool = False):
        """Initialize the FormFactorLocalDB with element and magnetic properties.
        
        Args:
            element (str): Chemical symbol of the element (e.g., 'Fe', 'O').
            is_magnetic (bool, optional): Whether to load magnetic form factors. 
                                        Defaults to False.
        """
        super().__init__()
        self.element = element
        self.is_magnetic = is_magnetic
        self.read_data()

    def read_data(self):
        """Read form factor data from the local database.
        
        Loads form factor data from a text file based on the element name and
        magnetic properties. The data is stored in a pandas DataFrame with
        columns for energy (E), real part (f1), and imaginary part (f2).
        
        Raises:
            ValueError: If no element is specified.
            FileNotFoundError: If the form factor data file does not exist.
        """
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
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        f1 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f1))
        f2 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f2))
        
        return f1, f2
    
    def get_all_formfactors(self, *args) -> pd.DataFrame:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        return self.ff_data
    