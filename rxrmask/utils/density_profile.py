"""Density profile utilities for RXR-Mask.

This module provides functions for calculating atomic density profiles in multilayer
structures, including the effects of interface roughness.
"""

import numpy as np
from scipy.special import erf


def step_function(z, zi):
    """Step function for sharp interface transitions.
    
    Args:
        z (np.ndarray): Array of spatial positions.
        zi (float): Position of the interface transition.
        
    Returns:
        np.ndarray: Array with values 0.0 for z < zi and 2.0 for z >= zi.
    """
    return np.where(z >= zi, 2.0, 0.0)

def density_profile(thicknesses, densities, roughness, step=0.1):
    """Calculate density profiles for multilayer structures with interface roughness.
    
    Args:
        thicknesses (list): List of layer thicknesses in Angstroms.
        densities (dict): Dictionary mapping element names to density arrays.
                         Each array contains density values for each layer.
        roughness (dict): Dictionary mapping element names to roughness arrays.
                         Each array contains roughness values for each interface.
        step (float, optional): Spatial resolution for the profile in Angstroms.
                               Defaults to 0.1.
                               
    Returns:
        tuple: A 2-tuple containing:
            - z (np.ndarray): Array of spatial positions in Angstroms.
            - dens (dict): Dictionary mapping element names to density profile arrays.
            
    Example:
        >>> thicknesses = [10.0, 20.0, 15.0]  # Layer thicknesses
        >>> densities = {'Fe': [0.0, 5.0, 3.0, 0.0]}  # Fe density in each region
        >>> roughness = {'Fe': [2.0, 1.5, 2.5]}  # Interface roughness
        >>> z, profiles = density_profile(thicknesses, densities, roughness)
    """
    pos = np.cumsum(thicknesses)

    max_roughness = max(roughness.values(), key=lambda x: x[-1])[-1]
    z = np.arange(0, pos[-1] + max_roughness*1.5, step)
    dens = {}

    for name, rho in densities.items():
        profile = np.full_like(z, rho[0], dtype=float)

        for i in range(len(thicknesses)):
            drho = rho[i+1] - rho[i]
            rough = roughness[name][i]

            if rough == 0:
                profile += 0.5 * (drho) * step_function(z, pos[i])
            else:
                profile += (drho/2) * (erf((z-pos[i])/(rough/np.sqrt(2))) + 1)

        dens[name] = profile
    return z, dens

def z_density_profile(z_eval, thicknesses, densities, roughness):
    """Calculate density profiles at specific spatial positions.
    
    Computes atomic density profiles at user-specified spatial positions,
    rather than on a regular grid. This function is useful when density
    values are needed only at specific locations, such as for fitting
    or when interfacing with other calculations.
    
    The calculation method is identical to density_profile() but evaluates
    the density functions only at the provided z positions, making it more
    efficient for sparse sampling.
    
    Args:
        z_eval (array-like): Array of spatial positions where densities should be evaluated.
        thicknesses (list): List of layer thicknesses in Angstroms.
        densities (dict): Dictionary mapping element names to density arrays.
                         Each array contains density values for each layer.
        roughness (dict): Dictionary mapping element names to roughness arrays.
                         Each array contains roughness values for each interface.
                         
    Returns:
        tuple: A 2-tuple containing:
            - z (np.ndarray): Array of input spatial positions (converted to numpy array).
            - dens (dict): Dictionary mapping element names to density arrays
                          evaluated at the specified positions.
                          
    Example:
        >>> z_points = [5.0, 15.0, 25.0]  # Specific positions of interest
        >>> z, profiles = z_density_profile(z_points, thicknesses, densities, roughness)
    """
    pos = np.cumsum(thicknesses)

    z = np.array(z_eval)
    dens = {}

    for name, rho in densities.items():
        profile = np.full_like(z, rho[0], dtype=float)

        for i in range(len(thicknesses)):
            drho = rho[i+1] - rho[i]
            rough = roughness[name][i]

            if rough == 0:
                profile += 0.5 * (drho) * step_function(z, pos[i])
            else:
                profile += (drho/2) * (erf((z-pos[i])/(rough/np.sqrt(2))) + 1)

        dens[name] = profile
    return z, dens
