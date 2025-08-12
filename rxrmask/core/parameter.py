from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import json


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
    id: int = 0
    name: str = ""

    lower: float | None = None
    upper: float | None = None
    fit: bool = False

    def __post_init__(self):
        factor = 0.2
        if self.lower is not None:
            self.lower = self.value * (1 - factor) if self.value != 0.0 else 0.0
        if self.upper is not None:
            self.upper = self.value * (1 + factor)

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
        if value is None:
            raise ValueError("Parameter value cannot be None.")
        if not isinstance(value, (int, float, str, complex)):
            raise TypeError("Parameter value must be an int, float, str, or complex number.")
        self.value = value


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

    def convert_to_parameter(self) -> Parameter:
        """Convert to regular Parameter."""
        return Parameter(
            id=self.id,
            name=self.name,
            value=self.get(),
            lower=self.lower,
            upper=self.upper,
            fit=self.fit,
        )


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
        param = Parameter(id=id, name=name, value=value, fit=fit)
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
        p = DerivedParameter(id=id, name=name, value=0.0, fit=False, update_func=update_func)
        self.derived_parameters.append(p)
        return p

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

    def save(self, path: str) -> None:
        """Save parameters to JSON file."""
        data = []
        for p in self.parameters:
            data.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "value": p.get(),
                    "min_value": p.lower,
                    "max_value": p.upper,
                    "fit": p.fit,
                }
            )
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, path: str) -> None:
        """Load parameters from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.parameters.clear()
        for entry in data:
            param = Parameter(
                id=entry["id"],
                name=entry["name"],
                value=entry["value"],
                lower=entry.get("min_value"),
                upper=entry.get("max_value"),
                fit=entry.get("fit", False),
            )
            self.parameters.append(param)
