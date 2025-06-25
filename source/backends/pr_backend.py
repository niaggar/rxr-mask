import numpy as np
import Pythonreflectivity as pr
from source.core.structure import Structure


HC_EV_ANGSTROM = 12398.41984


def _to_pr_structure(stack: Structure, E_eV: float):
    S = pr.Generate_structure(len(stack.layers))
    for i, L in enumerate(stack.layers):
        n = L.get_n(E_eV)
        S[i].seteps(n ** 2)
        S[i].setd(L.get_thickness_angstrom())
    return S


def reflectivity(stack: Structure, qz: np.ndarray, E_eV: float):
    S = _to_pr_structure(stack, E_eV)
    wavelength = HC_EV_ANGSTROM / E_eV  # Å
    theta_deg = np.degrees(np.arcsin(qz * wavelength / (4 * np.pi)))

    R_sigma, R_pi = pr.Reflectivity(S, theta_deg, wavelength)
    return R_sigma, R_pi
