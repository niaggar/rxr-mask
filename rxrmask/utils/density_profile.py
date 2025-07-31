"""Density profile utilities for RXR-Mask.

This module provides functions for calculating atomic density profiles in multilayer
structures, including the effects of interface roughness.
"""

import numpy as np
from scipy.special import erf

def _calculate_single_element_profile(z, layer_positions, densities, roughnesses):    
    profile = np.full_like(z, densities[0], dtype=float)
    
    for i in range(len(layer_positions) - 1):
        rho_current = densities[i]
        rho_next = densities[i + 1]
        
        density_change = rho_next - rho_current
        if density_change == 0:
            continue
            
        interface_position = layer_positions[i + 1]
        roughness = roughnesses[i]
        
        if roughness == 0:
            profile += 0.5 * density_change * np.where(z >= interface_position, 2.0, 0.0)
        else:
            sigma = roughness / np.sqrt(2)
            profile += (density_change / 2) * (erf((z - interface_position) / sigma) + 1)
    
    return profile

def get_density_profile_from_element_data(element_data, layer_thickness_params, atoms, step: float = 0.1):
    layer_thicknesses = [param.get() for param in layer_thickness_params]
    layer_positions = np.cumsum([0.0] + layer_thicknesses)
    
    max_roughness = 0.0
    for data in element_data.values():
        for roughness_param in data['roughness_params']:
            if roughness_param is not None:
                max_roughness = max(max_roughness, roughness_param.get())
    
    total_thickness = sum(layer_thicknesses)
    z = np.arange(0, total_thickness + 15 + step, step)
    
    density_profile = {}
    magnetic_density_profile = {}
    
    for element_name, data in element_data.items():
        densities = [param.get() if param is not None else 0.0 for param in data['density_params']]
        magnetic_densities = [param.get() if param is not None else 0.0 for param in data['magnetic_density_params']]
        roughnesses = [param.get() if param is not None else 0.0 for param in data['roughness_params']]
        
        density_profile[element_name] = _calculate_single_element_profile(
            z, layer_positions, densities, roughnesses
        )
        
        magnetic_density_profile[element_name] = _calculate_single_element_profile(
            z, layer_positions, magnetic_densities, roughnesses
        )
    
    return z, density_profile, magnetic_density_profile, atoms



