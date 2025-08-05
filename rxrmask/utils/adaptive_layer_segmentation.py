"""Adaptive layer segmentation for efficient reflectivity calculations.

Provides functions to group similar layers for computational optimization.
"""

from rxrmask.core.layer import Layer
from typing import List
import numpy as np


def compute_adaptive_layer_segmentation(
    index_of_refraction: np.ndarray,
    magnetic_optical_constants: np.ndarray,
    precision: float = 1e-6,
    als: bool = False,
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
    if als is False:
        return list(range(len(index_of_refraction)))
    
    indices = []
    idx_a = 0

    for idx_b in range(1, len(index_of_refraction)):
        eps_a = index_of_refraction[idx_a]
        eps_b = index_of_refraction[idx_b]
        q_a = magnetic_optical_constants[idx_a]
        q_b = magnetic_optical_constants[idx_b]
        
        delta_eps = abs(eps_a - eps_b)
        delta_q = abs(q_a - q_b)

        if delta_eps > precision or delta_q > precision:
            indices.append(idx_b)
            idx_a = idx_b

    if indices[-1] != len(index_of_refraction) - 1:
        indices.append(len(index_of_refraction) - 1)

    return indices
