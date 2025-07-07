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
    structure = _to_pr_structure(stack, E_eV)
    h = 4.135667696e-15  # Plank's constant eV*s
    c = 2.99792458e8  # speed of light m/s
    wavelength = h * c / (E_eV * 1e-10)  # wavelength of incoming x-ray
    theta_deg = np.arcsin(qz / E_eV / (0.001013546247)) * 180 / np.pi

    R_sigma, R_pi = pr.Reflectivity(structure, theta_deg, wavelength, MagneticCutoff=1e-20)
    return qz, R_sigma, R_pi

def energy_scan(stack: Structure, E_eVs: list[float], theta_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    structures = []
    for E_eV in E_eVs:
        S = _to_pr_structure(stack, E_eV)
        structures.append(S)
    
    h = 4.135667696e-15  # Plank's constant eV*s
    c = 2.99792458e8  # speed of light m/s

    E_np = np.array(E_eVs)
    wavelengths = h * c / (E_np * 1e-10)

    R_sigma_all = []
    R_pi_all = []
    for i, E_eV in enumerate(E_eVs):
        R_sigma, R_pi = pr.Reflectivity(structures[i], theta_deg, wavelengths[i], MagneticCutoff=1e-20)
        R_sigma_all.append(R_sigma)
        R_pi_all.append(R_pi)

    R_sigma_all = np.array(R_sigma_all)
    R_pi_all = np.array(R_pi_all)
    return E_np, R_sigma_all, R_pi_all