import numpy as np
import Pythonreflectivity as pr
from source.core.compound import Compound
from source.core.structure import Structure
from source.core.layer import Layer

from joblib import Parallel, delayed, parallel_backend


H_CONST = 4.135667696e-15 # Planck [eV·s]
C_CONST = 2.99792458e8 # luz  [m/s]
QZ_SCALE = 0.001013546247 # factor qz→theta
HC_EV_ANGSTROM = 12398.41984


def _to_pr_structure(stack: Structure, E_eV: float):
    S = pr.Generate_structure(stack.n_layers)

    for i, layer in enumerate(stack.layers):
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected a Layer instance at index {i} in the stack.")

        n = layer.get_index_of_refraction(E_eV)
        S[i].seteps(n ** 2)
        S[i].setd(layer.thickness)

    return S

def reflectivity(stack: Structure, qz: np.ndarray, E_eV: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    structure = _to_pr_structure(stack, E_eV)
    h = 4.135667696e-15  # Plank's constant eV*s
    c = 2.99792458e8  # speed of light m/s
    wavelength = h * c / (E_eV * 1e-10)  # wavelength of incoming x-ray
    theta_deg = np.arcsin(qz / E_eV / (0.001013546247)) * 180 / np.pi

    R_sigma, R_pi = pr.Reflectivity(structure, theta_deg, wavelength, MagneticCutoff=1e-20)
    return qz, R_sigma, R_pi



def _worker_process(stack, qz_i, E_eV):
    """Worker de multiprocessing: genera su propia Structure."""
    structure = _to_pr_structure(stack, E_eV)
    wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
    theta = np.arcsin(qz_i / E_eV / QZ_SCALE) * 180/np.pi
    R_sigma, R_pi = pr.Reflectivity(structure, theta, wavelength, MagneticCutoff=1e-20)
    return qz_i, R_sigma, R_pi

def _worker_thread(structure, qz_i, E_eV):
    wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
    theta = np.arcsin(qz_i / E_eV / QZ_SCALE) * 180/np.pi
    R_sigma, R_pi = pr.Reflectivity(structure, theta, wavelength, MagneticCutoff=1e-20)
    return qz_i, R_sigma, R_pi

def reflectivity_parallel(stack, qz: np.ndarray, E_eV: float, n_jobs: int = -1, use_threads: bool = False, verbose: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if use_threads:
        structure = _to_pr_structure(stack, E_eV)
        backend = 'threading'
        tasks = [delayed(_worker_thread)(structure, qzi, E_eV) for qzi in qz]
    else:
        backend = 'loky'
        tasks = [delayed(_worker_process)(stack, qzi, E_eV) for qzi in qz]

    with parallel_backend(backend, n_jobs=n_jobs):
        results = Parallel(verbose=verbose)(tasks)

    qz_out, R_sigma, R_pi = zip(*results)
    return np.array(qz_out), np.array(R_sigma), np.array(R_pi)




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



def _energy_scan_process(stack, E_eV, theta_deg):
    structure = _to_pr_structure(stack, E_eV)
    wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
    R_sigma, R_pi = pr.Reflectivity(
        structure,
        theta_deg,
        wavelength,
        MagneticCutoff=1e-20
    )
    return R_sigma, R_pi

def _energy_scan_thread(structure, wavelength, theta_deg):
    R_sigma, R_pi = pr.Reflectivity(
        structure,
        theta_deg,
        wavelength,
        MagneticCutoff=1e-20
    )
    return R_sigma, R_pi

def energy_scan_parallel(stack: Structure, E_eVs: list[float], theta_deg: float, n_jobs: int = -1, use_threads: bool = False, verbose: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E_np = np.array(E_eVs)
    wavelengths = H_CONST * C_CONST / (E_np * 1e-10)

    if use_threads:
        structures = [_to_pr_structure(stack, E) for E in E_eVs]
        backend = 'threading'
        tasks = [
            delayed(_energy_scan_thread)(structures[i], wavelengths[i], theta_deg) for i in range(len(E_eVs))
        ]
    else:
        backend = 'loky'
        tasks = [
            delayed(_energy_scan_process)(stack, E, theta_deg) for E in E_eVs
        ]

    with parallel_backend(backend, n_jobs=n_jobs):
        results = Parallel(verbose=verbose)(tasks)

    R_sigma_all, R_pi_all = zip(*results)
    return E_np, np.array(R_sigma_all), np.array(R_pi_all)
