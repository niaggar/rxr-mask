"""Parameter module for RXR-Mask.

This module provides classes for managing parameters in X-ray reflectometry
calculations and fitting procedures. It includes basic parameters for storing
values and fit parameters for optimization routines.

The parameter system supports type validation, value constraints, and parameter
containers for organizing collections of parameters used in modeling and fitting.
"""

from dataclasses import dataclass, field

@dataclass
class Parameter:
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
    id: int
    name: str
    value: None = None

    def __post_init__(self):
        """Post-initialization validation of parameter value.
        
        Validates that the parameter value is not None and is of a supported type.
        
        Raises:
            ValueError: If the parameter value is None.
            TypeError: If the parameter value is not int, float, str, or complex.
        """
        if self.value is None:
            raise ValueError("Parameter value cannot be None.")
        if not isinstance(self.value, (int, float, str, complex)):
            raise TypeError("Parameter value must be an int, float, str, or complex number.")
        
    def get(self, prray=None) -> None:
        """Get the parameter value.
        
        Args:
            prray: Parameter array (unused in base Parameter class).
            
        Returns:
            The parameter value.
            
        Raises:
            ValueError: If the parameter value has not been set.
        """
        if self.value is None:
            raise ValueError("Parameter value has not been set.")
        return self.value

    def set(self, value: None) -> None:
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
class FitParameter(Parameter):
    """Parameter class designed for optimization and fitting procedures.
    
    This class extends Parameter to support fitting operations by adding
    initial values, bounds constraints, and fit toggling capabilities.
    It's designed to work with optimization algorithms that require
    parameter bounds and initial guesses.
    
    Attributes:
        init_value: Initial value for fitting. Defaults to None.
        min_value: Minimum allowed value during fitting. Defaults to None.
        max_value: Maximum allowed value during fitting. Defaults to None.
    
    Inherits:
        All attributes and methods from Parameter class.
    """
    init_value: None = None
    min_value: None = None
    max_value: None = None

    def __post_init__(self):
        """Post-initialization validation for fit parameters.
        
        Validates the parameter and checks fit parameter requirements.
        Note: This method assumes a 'fit' attribute exists but it's not 
        defined in the class attributes.
        
        Raises:
            ValueError: If parameter value validation fails.
            TypeError: If parameter type validation fails or fit is not boolean.
        """
        super().__post_init__()
        if not isinstance(self.fit, bool):
            raise TypeError("Fit parameter must be a boolean value.")
    
    def toggle_fit(self) -> None:
        """Toggle the fitting state of the parameter.
        
        Switches the fit flag between True and False, enabling or disabling
        the parameter for fitting procedures.
        """
        self.fit = not self.fit
    
    def set_fit_range(self, init_value: None, min_value: None, max_value: None) -> None:
        """Set the fitting range and initial value for the parameter.
        
        Configures the parameter for fitting by setting initial value and bounds.
        The current parameter value is updated to the initial value.
        
        Args:
            init_value: Initial value for fitting. Must be int or float if not None.
            min_value: Minimum allowed value during fitting.
            max_value: Maximum allowed value during fitting.
            
        Raises:
            TypeError: If init_value is not None and not int or float.
        """
        if init_value is not None and not isinstance(init_value, (int, float)):
            raise TypeError("Initial value must be an int or float.")
        
        self.init_value = init_value
        self.min_value = min_value
        self.max_value = max_value
        self.value = init_value

    def get(self, prray=None) -> None:
        """Get the parameter value from a parameter array.
        
        Retrieves the parameter value from the provided parameter array
        using the parameter's ID as an index. This is used during fitting
        when parameters are stored in arrays.
        
        Args:
            prray: Parameter array containing current values during fitting.
            
        Returns:
            The parameter value from the array at index self.id.
        """
        return prray[self.id] # type: ignore

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
    parameters: list[Parameter] = field(default_factory=list)
    fit_parameters: list[FitParameter] = field(default_factory=list)

    def new_fit_parameter(self, name: str, init_value: None = None, min_value: None = None, max_value: None = None) -> FitParameter:
        """Create a new fit parameter and add it to the container.
        
        Factory method that creates a new FitParameter with automatic ID assignment
        and adds it to the fit_parameters list.
        
        Args:
            name (str): Human-readable name for the parameter.
            init_value: Initial value for fitting. Defaults to None.
            min_value: Minimum allowed value during fitting. Defaults to None.
            max_value: Maximum allowed value during fitting. Defaults to None.
            
        Returns:
            FitParameter: The newly created fit parameter.
        """
        id = len(self.fit_parameters)

        param = FitParameter(id=id, name=name, value=init_value)
        param.set_fit_range(init_value, min_value, max_value)
        self.fit_parameters.append(param)
        return param
    
    def new_parameter(self, name: str, value: None) -> Parameter:
        """Create a new regular parameter and add it to the container.
        
        Factory method that creates a new Parameter with automatic ID assignment
        and adds it to the parameters list.
        
        Args:
            name (str): Human-readable name for the parameter.
            value: The parameter value.
            
        Returns:
            Parameter: The newly created parameter.
        """
        id = len(self.fit_parameters)
        param = Parameter(id=id, name=name, value=value)
        self.parameters.append(param)
        return param
    
    def get_vector(self) -> list[None]:
        """Extract initial values from fit parameters that are enabled for fitting.
        
        Creates a vector of initial values from all fit parameters that have
        their fit flag set to True. This is typically used to initialize
        optimization algorithms.
        
        Returns:
            list: List of initial values from enabled fit parameters.
        """
        return [param.init_value for param in self.fit_parameters if param.fit]
