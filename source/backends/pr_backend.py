import numpy as np
import Pythonreflectivity as pr
from source.core.compound import Compound
from source.core.structure import Structure


HC_EV_ANGSTROM = 12398.41984


def _to_pr_structure(stack: Structure, E_eV: float):
    S = pr.Generate_structure(stack.n_layers)

    layer_offset = 0
    for compound in stack.compounds:
        if not isinstance(compound, Compound):
            raise TypeError("Expected a Compound instance in the stack.")
        
        for i_layer in range(compound.n_layers):
            n = compound.get_n_layer(E_eV, i_layer)
            thickness = compound.get_thickness_layer(i_layer)

            layer_index = layer_offset + i_layer
            
            S[layer_index].seteps(n ** 2)
            S[layer_index].setd(thickness)

        layer_offset += compound.n_layers

    return S

def reflectivity(stack: Structure, qz: np.ndarray, E_eV: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = _to_pr_structure(stack, E_eV)
    h = 4.135667696e-15  # Plank's constant eV*s
    c = 2.99792458e8  # speed of light m/s
    wavelength = h * c / (E_eV * 1e-10)  # wavelength of incoming x-ray
    theta_deg = np.arcsin(qz / E_eV / (0.001013546247)) * 180 / np.pi

    R_sigma, R_pi = pr.Reflectivity(S, theta_deg, wavelength, MagneticCutoff=1e-20)
    return qz, R_sigma, R_pi

