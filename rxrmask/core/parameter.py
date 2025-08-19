from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Parameter:
    """Parameter with value, constraints, and fitting capability.

    Attributes:
        value: Parameter value
        id: Unique identifier
        name: Human-readable name, also used to save/load
        lower: Minimum value constraint
        upper: Maximum value constraint
        fit: Enable for fitting/optimization
    """

    value: float
    id: int
    name: str
    lower: float | None
    upper: float | None
    fit: bool

    def __post_init__(self):
        factor = 0.2

        if self.lower is None:
            self.lower = self.value * (1 - factor) if self.value != 0.0 else 0.0
        if self.upper is None:
            self.upper = self.value * (1 + factor)
            if self.upper == 0.0:
                self.upper = factor

    def get(self) -> float:
        """Get parameter value from internal storage.

        Returns:
            Parameter value
        """
        return self.value

    def set(self, value: float) -> None:
        """Set parameter value with type validation.

        Args:
            value: New parameter value
        """
        self.value = value

    def set_bounds(self, lower: float, upper: float) -> None:
        """Set bounds for the parameter.

        Args:
            lower: Minimum value constraint
            upper: Maximum value constraint
        """
        if lower == upper:
            print("Warning: Lower and upper bounds are equal; parameter will be fixed.")
            self.lower, self.upper = lower, upper
            return

        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        self.lower = lower
        self.upper = upper


@dataclass
class DerivedParameter(Parameter):
    """Parameter with automatic update functionality.

    Attributes:
        update_func: Function to calculate new value
    """

    update_func: Optional[Callable[[], float]] = None

    def update(self):
        """Update parameter value using update function."""
        if self.update_func is None:
            raise RuntimeError("No update function defined for DerivedParameter.")
        self.value = self.update_func()

    def get(self) -> float:
        self.update()
        return self.value

    def set(self, value: float) -> None:
        print(f"Warning: Cannot set value for DerivedParameter '{self.name}'. It is auto-updating.")


@dataclass
class DependentParameter(Parameter):
    """Parameter that depends on another parameter.

    Attributes:
        independent: If True, this parameter can be set independently
        depends_on: The parameter this one depends on
        update_func: Function to calculate value if independent
        value: Initial value, can be updated by the update function
    """

    independent: bool = False
    depends_on: Parameter | None = None
    update_func: Optional[Callable[[], float]] = None

    def get(self) -> float:
        """Get value from the dependent parameter."""
        if self.independent:
            return self.value
        elif not self.independent and self.update_func is not None:
            self.value = self.update_func()
            return self.value
        elif not self.independent and self.depends_on is not None:
            return self.depends_on.get()
        else:
            raise RuntimeError("DependentParameter is not independent and has no valid dependency.")

    def set(self, value: float) -> None:
        """Set value for the dependent parameter."""
        if self.independent:
            self.value = value
        else:
            print(f"Warning: Cannot set value for DependentParameter '{self.name}'. It depends on another parameter.")
            self.value = value


@dataclass
class ParametersContainer:
    """Container for managing parameter collections and fitting workflows.

    Attributes:
        parameters: List of regular parameters
        derived_parameters: List of auto-updating parameters
    """

    parameters: list[Parameter] = field(default_factory=list)
    derived_parameters: list[DerivedParameter] = field(default_factory=list)

    def new_parameter(self, name: str, value: Any, fit: bool = False) -> Parameter:
        """Create and register a new parameter.

        Args:
            name: Parameter name
            value: Initial value
            fit: Enable for optimization

        Returns:
            Created Parameter object
        """
        id = len(self.parameters)
        param = Parameter(id=id, name=name, value=value, fit=fit, lower=None, upper=None)
        self.parameters.append(param)
        return param

    def register_derived(self, name: str, update_func: Callable[[], Any]) -> DerivedParameter:
        """Register a derived parameter with update function.

        Args:
            name: Parameter name
            update_func: Function to calculate value

        Returns:
            Created DerivedParameter object
        """
        id = len(self.derived_parameters)
        p = DerivedParameter(id=id, name=name, value=0.0, fit=False, update_func=update_func, lower=None, upper=None)
        self.derived_parameters.append(p)
        return p

    def register_param(self, param: Parameter):
        """Register an existing parameter object."""
        id = len(self.parameters)
        param.id = id
        self.parameters.append(param)

    def update_derived(self):
        """Update all derived parameters."""
        for p in self.derived_parameters:
            p.update()

    def get_fit_vector(self) -> list[float]:
        """Extract values of parameters marked for fitting."""
        return [float(p.get()) for p in self.parameters if p.fit]

    def set_fit_vector(self, values: list[float]):
        """Update fitting parameters from vector."""
        i = 0
        for p in self.parameters:
            if p.fit:
                p.set(values[i])
                i += 1
        if i != len(values):
            raise ValueError("Mismatch between fit parameters and values vector.")
        self.update_derived()
