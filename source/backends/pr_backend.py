import numpy as np
import Pythonreflectivity as pr
from source.core.structure import Structure


HC_EV_ANGSTROM = 12398.41984


def _to_pr_structure(stack: Structure, E_eV: float):
    total_layers = sum(comp.n_layers for comp in stack.compounds)
    S = pr.Generate_structure(total_layers)

    layer_offset = 0
    for compound in stack.compounds:
        for i_layer in range(compound.n_layers):
            n = compound.get_n_layer(E_eV, i_layer)
            thickness = compound.get_thickness_layer(i_layer)
            layer_index = layer_offset + i_layer
            S[layer_index].seteps(n ** 2)
            S[layer_index].setd(thickness)
        layer_offset += compound.n_layers

    return S

def reflectivity(stack: Structure, qz: np.ndarray, E_eV: float):
    S = _to_pr_structure(stack, E_eV)
    wavelength = HC_EV_ANGSTROM / E_eV  # Å
    theta_deg = np.degrees(np.arcsin(qz * wavelength / (4 * np.pi)))

    R_sigma, R_pi = pr.Reflectivity(S, theta_deg, wavelength)
    return R_sigma, R_pi

