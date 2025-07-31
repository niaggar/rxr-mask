"""Adaptive layer segmentation for efficient reflectivity calculations.

Provides functions to group similar layers for computational optimization.
"""

from rxrmask.core.layer import Layer
from typing import List


def compute_adaptive_layer_segmentation(
    layers: List[Layer],
    energy_eV: float,
    precision: float = 1e-6
) -> List[int]:
    """Adaptive layer segmentation based on optical constant variation.

    Groups layers with similar optical properties to reduce computation.

    Args:
        layers: List of layers with precomputed optical properties.
        energy_eV: Energy at which optical constants were computed.
        precision: Threshold for optical constant variation.

    Returns:
        List[int]: Layer boundary indices for segmentation.
    """
    if not layers:
        return []

    indices = [0]
    idx_a = 0

    for idx_b in range(1, len(layers)):
        layer_a = layers[idx_a]
        layer_b = layers[idx_b]

        # Safety check
        if layer_a.precomputed_energy != energy_eV or layer_b.precomputed_energy != energy_eV:
            raise ValueError("All layers must have precomputed values at the target energy.")

        eps_a = layer_a.precomputed_index_of_refraction
        eps_b = layer_b.precomputed_index_of_refraction
        q_a = layer_a.precomputed_magnetic_optical_constant
        q_b = layer_b.precomputed_magnetic_optical_constant

        delta_var = abs(eps_b.real - eps_a.real) # type: ignore
        beta_var = abs(eps_b.imag - eps_a.imag) # type: ignore
        delta_m_var = abs(q_b.real - q_a.real) # type: ignore
        beta_m_var = abs(q_b.imag - q_a.imag) # type: ignore

        if any(var > precision for var in [delta_var, beta_var, delta_m_var, beta_m_var]):
            indices.append(idx_b)
            idx_a = idx_b

    if indices[-1] != len(layers) - 1:
        indices.append(len(layers) - 1)

    return indices
