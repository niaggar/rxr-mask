from dataclasses import dataclass, field

@dataclass
class Parameter:
    id: int
    name: str
    value: None = None

    def __post_init__(self):
        if self.value is None:
            raise ValueError("Parameter value cannot be None.")
        if not isinstance(self.value, (int, float, str, complex)):
            raise TypeError("Parameter value must be an int, float, str, or complex number.")
        
    def get(self, prray=None) -> None:
        if self.value is None:
            raise ValueError("Parameter value has not been set.")
        return self.value

    def set(self, value: None) -> None:
        if value is None:
            raise ValueError("Parameter value cannot be None.")
        if not isinstance(value, (int, float, str, complex)):
            raise TypeError("Parameter value must be an int, float, str, or complex number.")
        self.value = value

@dataclass
class FitParameter(Parameter):
    init_value: None = None
    min_value: None = None
    max_value: None = None

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.fit, bool):
            raise TypeError("Fit parameter must be a boolean value.")
    
    def toggle_fit(self) -> None:
        self.fit = not self.fit
    
    def set_fit_range(self, init_value: None, min_value: None, max_value: None) -> None:
        if init_value is not None and not isinstance(init_value, (int, float)):
            raise TypeError("Initial value must be an int or float.")
        
        self.init_value = init_value
        self.min_value = min_value
        self.max_value = max_value
        self.value = init_value

    def get(self, prray=None) -> None:
        return prray[self.id] # type: ignore

@dataclass
class ParametersContainer:
    parameters: list[Parameter] = field(default_factory=list)
    fit_parameters: list[FitParameter] = field(default_factory=list)

    def new_fit_parameter(self, name: str, init_value: None = None, min_value: None = None, max_value: None = None) -> FitParameter:
        id = len(self.fit_parameters)

        param = FitParameter(id=id, name=name, value=init_value)
        param.set_fit_range(init_value, min_value, max_value)
        self.fit_parameters.append(param)
        return param
    
    def new_parameter(self, name: str, value: None) -> Parameter:
        id = len(self.fit_parameters)
        param = Parameter(id=id, name=name, value=value)
        self.parameters.append(param)
        return param
    
    def get_vector(self) -> list[None]:
        return [param.init_value for param in self.fit_parameters if param.fit]
