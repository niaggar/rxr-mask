import numpy as np
from scipy.special import erf


def _calculate_single_element_profile(z, layer_positions, densities, roughnesses):
    """Calculate density profile for single element with interface roughness."""
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


def get_density_profile_from_element_data(element_data, step: float = 0.1):
    """Calculate density profiles from element data and layer parameters.
    
    Args:
        element_data: Element data dictionary with parameters.
        step: Depth step size for profile calculation.
        
    Returns:
        tuple: (z_positions, density_profiles, magnetic_density_profiles).
    """

    total_thickness = 0
    for data in element_data.values():
        temp_thick = sum(param.get() for param in data['thickness_params'])
        if temp_thick > total_thickness:
            total_thickness = temp_thick

    z = np.arange(0, total_thickness + 15 + step, step)

    density_profile = {}
    magnetic_density_profile = {}

    for element_name, data in element_data.items():
        densities = [param.get() if param is not None else 0.0 for param in data['density_params']]
        magnetic_densities = [param.get() if param is not None else 0.0 for param in data['magnetic_density_params']]
        roughnesses = [param.get() if param is not None else 0.0 for param in data['roughness_params']]
        thickness = [param.get() if param is not None else 0.0 for param in data['thickness_params']]
        layer_positions = np.cumsum([0.0] + thickness)

        density_profile[element_name] = _calculate_single_element_profile(
            z, layer_positions, densities, roughnesses
        )

        magnetic_density_profile[element_name] = _calculate_single_element_profile(
            z, layer_positions, magnetic_densities, roughnesses
        )

    return z, density_profile, magnetic_density_profile
