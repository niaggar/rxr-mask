import numpy as np
from typing import List


def compute_adaptive_layer_segmentation(
    index_of_refraction: np.ndarray,
    magnetic_optical_constants: np.ndarray,
    precision: float = 1e-6,
    als: bool = False,
) -> List[int]:
    """Adaptive layer segmentation based on optical constant variation.

    Args:
        index_of_refraction: Index of refraction array
        magnetic_optical_constants: Magnetic optical constants
        precision: Threshold for optical constant variation
        als: Enable adaptive layer segmentation

    Returns:
        Layer boundary indices for segmentation
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
