from rxrmask.backends.reflectivitybackend import ReflectivityBackend
from rxrmask.core.parameter import Parameter, ParametersContainer
from rxrmask.core.structure import Structure
from rxrmask.core.reflectivitydata import EnergyScanData, ReflectivityData

import numpy as np

from rxrmask.optimization.optimization import Optimizer

class RXRModel:
    structure: Structure
    parameters_container: ParametersContainer
    backend: ReflectivityBackend
    R_scale: Parameter[float]
    R_offset: Parameter[float]

    def __init__(self, structure: Structure, parameters_container: ParametersContainer):
        self.structure = structure
        self.parameters_container = parameters_container
        self.R_scale = self.parameters_container.new_parameter("R_scale", 1.0, fit=True)
        self.R_offset = self.parameters_container.new_parameter("R_offset", 0.0, fit=True)

    def set_reflectivity_backend(self, backend: ReflectivityBackend) -> None:
        """Set the computational backend for the structure."""
        self.backend = backend
    
    def compute_reflectivity(self, q, e: float) -> ReflectivityData:
        """Compute the reflectivity using the selected backend."""
        if not hasattr(self, 'backend'):
            raise ValueError("Reflectivity backend is not set.")

        r_data = self.backend.compute_reflectivity(self.structure, q, e)
        r_data = self.apply_scaling(r_data)
        return r_data
    
    def compute_energy_scan(self, e, theta: float) -> EnergyScanData:
        """Compute the energy scan reflectivity using the selected backend."""
        if not hasattr(self, 'backend'):
            raise ValueError("Reflectivity backend is not set.")

        r_data = self.backend.compute_energy_scan(self.structure, e, theta)
        r_data = self.apply_scaling(r_data)
        return r_data
    
    def apply_scaling(self, r_data):
        """Apply scaling to the reflectivity data."""
        scale = self.R_scale.get()
        offset = self.R_offset.get()
        
        r_data.R_s = np.log(r_data.R_s * scale + offset)
        r_data.R_p = np.log(r_data.R_p * scale + offset)
        return r_data


class RXRFitter:
    model: RXRModel
    experiment: ReflectivityData
    optimizer: Optimizer

    def __init__(self, model: RXRModel, experiment: ReflectivityData, optimizer: Optimizer):
        self.model = model
        self.experiment = experiment
        self.optimizer = optimizer

    def fit(self, initial_params, bounds):
        return self.optimizer.minimize(initial_params, self.loss_function, bounds)

    def loss_function(self, fit_params):
        self.model.parameters_container.set_fit_vector(fit_params)
        
        qz = self.experiment.qz
        energy = self.experiment.energy
        simulated = self.model.compute_reflectivity(qz, energy)

        if len(simulated.R_s) > 1:
            return np.sum((self.experiment.R_s - simulated.R_s)**2)
        if len(simulated.R_p) > 1:
            return np.sum((self.experiment.R_p - simulated.R_p)**2)

        raise ValueError("Simulated reflectivity data is empty or has invalid shape.")
