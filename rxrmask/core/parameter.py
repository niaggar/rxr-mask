"""Parameter module for RXR-Mask.

This module provides classes for managing parameters in X-ray reflectometry
calculations and fitting procedures. It includes basic parameters for storing
values and fit parameters for optimization routines.

The parameter system supports type validation, value constraints, and parameter
containers for organizing collections of parameters used in modeling and fitting.
"""

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any, Callable, Optional
import json

T = TypeVar('T')

@dataclass
class Parameter(Generic[T]):
    """Basic parameter class for storing named values.
    
    This class represents a parameter with an identifier, name, and value.
    It provides type validation to ensure values are of supported types
    (int, float, str, or complex).
    
    Attributes:
        id (int): Unique identifier for the parameter.
        name (str): Human-readable name of the parameter.
        value: The parameter value. Must be int, float, str, or complex. Defaults to None.
    
    Raises:
        ValueError: If value is None after initialization.
        TypeError: If value is not of a supported type.
    """
    value: T
    id: int = 0
    name: str = ""
    
    min_value: T | None = None  # Optional minimum value constraint
    max_value: T | None = None  # Optional maximum value constraint
    fit: bool = False  # Flag to indicate if the parameter is enabled for fitting
        
    def get(self, prray=None) -> T:
        """Get the parameter value.
        
        Args:
            prray: Parameter array (unused in base Parameter class).
            
        Returns:
            The parameter value.
            
        Raises:
            ValueError: If the parameter value has not been set.
        """
        if prray is None:
            if self.value is None:
                raise ValueError("Parameter value has not been set.")
            return self.value
        else:
            return prray[self.id] if self.id < len(prray) else None # type: ignore

    def set(self, value: T) -> None:
        """Set the parameter value.
        
        Args:
            value: New value for the parameter. Must be int, float, str, or complex.
            
        Raises:
            ValueError: If value is None.
            TypeError: If value is not of a supported type.
        """
        if value is None:
            raise ValueError("Parameter value cannot be None.")
        if not isinstance(value, (int, float, str, complex)):
            raise TypeError("Parameter value must be an int, float, str, or complex number.")
        self.value = value

@dataclass
class DerivedParameter(Parameter[T]):
    update_func: Optional[Callable[[], T]] = None  # Function to update the parameter value
    
    def update(self):
        if self.update_func is None:
            raise RuntimeError("No update function defined for DerivedParameter.")
        self.value = self.update_func()
        
@dataclass
class ParametersContainer:
    """Container for managing collections of parameters and fit parameters.
    
    This class provides a centralized way to manage both regular parameters
    and fit parameters used in X-ray reflectometry modeling and fitting.
    It includes factory methods for creating new parameters and utilities
    for extracting parameter vectors for optimization algorithms.
    
    Attributes:
        parameters (list[Parameter]): List of regular parameters.
        fit_parameters (list[FitParameter]): List of parameters available for fitting.
    """
    parameters: list[Parameter[Any]] = field(default_factory=list)
    derived_parameters: list[DerivedParameter[Any]] = field(default_factory=list)

    def new_parameter(self, name: str, value: Any, fit: bool = False) -> Parameter:
        """Create a new regular parameter and add it to the container.
        
        Factory method that creates a new Parameter with automatic ID assignment
        and adds it to the parameters list.
        
        Args:
            name (str): Human-readable name for the parameter.
            value: The parameter value.
            
        Returns:
            Parameter: The newly created parameter.
        """
        id = len(self.parameters)
        param = Parameter(id=id, name=name, value=value, fit=fit)
        self.parameters.append(param)
        return param
    
    def register_derived(self, name: str, update_func: Callable[[], Any]) -> DerivedParameter:
        id = len(self.derived_parameters)
        p = DerivedParameter(id=id, name=name, value=None, fit=False, update_func=update_func)
        self.derived_parameters.append(p)
        return p
    
    def update_derived(self):
        for p in self.derived_parameters:
            p.update()
    
    def get_fit_vector(self) -> list[float]:
        """Extrae los valores de los parámetros que están marcados como 'fit'."""
        return [float(p.value) for p in self.parameters if p.fit]

    def set_fit_vector(self, values: list[float]):
        """Actualiza los parámetros que están marcados como 'fit' desde un vector externo."""
        i = 0
        for p in self.parameters:
            if p.fit:
                p.set(values[i])
                i += 1
        if i != len(values):
            raise ValueError("Mismatch between fit parameters and values vector.")
        self.update_derived()
        
    def save(self, path: str) -> None:
        data = []
        for p in self.parameters:
            data.append({
                "id": p.id,
                "name": p.name,
                "value": p.value,
                "min_value": p.min_value,
                "max_value": p.max_value,
                "fit": p.fit
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)

        self.parameters.clear()
        for entry in data:
            param = Parameter(
                id=entry["id"],
                name=entry["name"],
                value=entry["value"],
                min_value=entry.get("min_value"),
                max_value=entry.get("max_value"),
                fit=entry.get("fit", False)
            )
            self.parameters.append(param)
    