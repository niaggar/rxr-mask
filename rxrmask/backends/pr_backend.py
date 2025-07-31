"""Pythonreflectivity backend for RXR-Mask.

This module provides an interface to the Pythonreflectivity library for calculating
X-ray reflectivity from multilayer structures. It includes both sequential and parallel
implementations for reflectivity calculations and energy scans.

The module handles the conversion between RXR-Mask structures and Pythonreflectivity
format, and provides optimized parallel computation capabilities for large datasets.
"""

from rxrmask.core.compound import Compound
from rxrmask.core.structure import Structure
from rxrmask.core.layer import Layer
from rxrmask.utils import compute_adaptive_layer_segmentation

import numpy as np
import Pythonreflectivity as pr
from joblib import Parallel, delayed, parallel_backend


H_CONST = 4.135667696e-15  # Planck constant [eV·s]
C_CONST = 2.99792458e8  # Speed of light [m/s]
QZ_SCALE = 0.001013546247  # Conversion factor qz→theta
HC_EV_ANGSTROM = 12398.41984  # hc constant [eV·Å]


def _compound_lookup_table(stack: Structure) -> list[tuple[float, Compound]]:
    """Build a sorted table of (z_top, compound) to map depth to compound."""
    z = 0.0
    table = []
    for compound in stack.compounds:
        if compound is not None:
            z += compound.thickness.get()
            table.append((z, compound))
    return table

def _find_compound_at_z(z: float, table: list[tuple[float, Compound]]) -> Compound:
    """Find the compound corresponding to a given z position."""
    for z_top, comp in table:
        if z <= z_top:
            return comp
    return table[-1][1]

def _to_pr_structure_from_segments(
    layers: list[Layer],
    indices: list[int],
    energy_eV: float,
    compound_map: list[tuple[float, Compound]]
):
    """
    Build PythonReflectivity structure from segments of layers defined by ALS indices.
    Each segment is represented by the first layer in the segment, but thickness is summed.
    """
    S = pr.Generate_structure(len(indices) - 1)
    z = 0.0

    for i in range(len(indices) - 1):
        i_start = indices[i]
        i_end = indices[i + 1]

        representative_layer = layers[i_start]
        segment_thickness = sum(layers[j].thickness for j in range(i_start, i_end))
        z += segment_thickness

        compound = _find_compound_at_z(z, compound_map)
        
        # print(f"Segment {i}: {compound.name} at z={z:.2f} Å, thickness={segment_thickness:.2f} Å")

        n = representative_layer.get_index_of_refraction(energy_eV)
        q = representative_layer.get_magnetic_optical_constant(energy_eV)
        eps = n**2
        eps_m = -1 * eps * q

        S[i].setmag(compound.magnetic_direction)
        S[i].seteps([eps, eps, eps, eps_m])
        if i != 0:
            S[i].setd(segment_thickness)

    return S



def reflectivity(stack: Structure, qz: np.ndarray, E_eV: float, precision: float = 1e-6):
    for layer in stack.layers:
        layer.compute(E_eV)

    compound_map = _compound_lookup_table(stack)
    
    indices = compute_adaptive_layer_segmentation(stack.layers, E_eV, precision)
    structure = _to_pr_structure_from_segments(stack.layers, indices, E_eV, compound_map)
    
    # _print_structure_info(structure)

    h = 4.135667696e-15  # Plank's constant eV*s
    c = 2.99792458e8  # speed of light m/s
    wavelength = h * c / (E_eV * 1e-10)  # wavelength of incoming x-ray
    theta_deg = np.arcsin(qz / E_eV / (0.001013546247)) * 180 / np.pi
    
    R_sigma, R_pi = pr.Reflectivity(structure, theta_deg, wavelength, MagneticCutoff=1e-20)
    
    return qz, R_sigma, R_pi


def _print_structure_info(structure: pr.Structure):
    print(f"Structure with {len(structure)} layers:")
    for i, layer in enumerate(structure):
        d = layer.d()
        sigma = layer.sigma()
        eps = layer.eps()
        mag = layer.mag()
        epsg = None
        try:
            epsg = layer.epsg()
        except AttributeError:
            pass
        print(f"Layer {i}: d={d:.2f} Å, sigma={sigma}, eps={eps}, mag={mag}, epsg={epsg}")





# def _worker_process(stack, qz_i, E_eV):
#     """Worker function for multiprocessing reflectivity calculations.
    
#     Generates its own structure and calculates reflectivity for a single qz value.
#     Used in parallel processing to avoid shared memory issues.
    
#     Args:
#         stack: The multilayer structure.
#         qz_i: Single momentum transfer value.
#         E_eV (float): X-ray energy in electron volts.
        
#     Returns:
#         tuple: (qz_i, R_sigma, R_pi) for the given qz value.
#     """
#     structure = _to_pr_structure(stack, E_eV)
#     wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
#     theta = np.arcsin(qz_i / E_eV / QZ_SCALE) * 180/np.pi
#     R_sigma, R_pi = pr.Reflectivity(structure, theta, wavelength, MagneticCutoff=1e-20)
#     return qz_i, R_sigma, R_pi

# def _worker_thread(structure, qz_i, E_eV):
#     """Worker function for threading reflectivity calculations.
    
#     Calculates reflectivity for a single qz value using a pre-generated structure.
#     Used in thread-based parallel processing where structures can be shared.
    
#     Args:
#         structure: Pre-generated Pythonreflectivity structure.
#         qz_i: Single momentum transfer value.
#         E_eV (float): X-ray energy in electron volts.
        
#     Returns:
#         tuple: (qz_i, R_sigma, R_pi) for the given qz value.
#     """
#     wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
#     theta = np.arcsin(qz_i / E_eV / QZ_SCALE) * 180/np.pi
#     R_sigma, R_pi = pr.Reflectivity(structure, theta, wavelength, MagneticCutoff=1e-20)
#     return qz_i, R_sigma, R_pi

# def reflectivity_parallel(stack, qz: np.ndarray, E_eV: float, n_jobs: int = -1, use_threads: bool = False, verbose: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Calculate X-ray reflectivity using parallel processing.
    
#     Computes reflectivity in parallel across multiple qz values using either
#     multiprocessing or threading for improved performance on large datasets.
    
#     Args:
#         stack: The multilayer structure.
#         qz (np.ndarray): Array of momentum transfer values in inverse Angstroms.
#         E_eV (float): X-ray energy in electron volts.
#         n_jobs (int, optional): Number of parallel jobs. -1 uses all cores. Defaults to -1.
#         use_threads (bool, optional): Use threading instead of multiprocessing. Defaults to False.
#         verbose (int, optional): Verbosity level for parallel execution. Defaults to 5.
        
#     Returns:
#         tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
#             - qz_out: Sorted momentum transfer array
#             - R_sigma: Reflectivity for sigma polarization
#             - R_pi: Reflectivity for pi polarization
#     """
#     if use_threads:
#         structure = _to_pr_structure(stack, E_eV)
#         backend = 'threading'
#         tasks = [delayed(_worker_thread)(structure, qzi, E_eV) for qzi in qz]
#     else:
#         backend = 'loky'
#         tasks = [delayed(_worker_process)(stack, qzi, E_eV) for qzi in qz]

#     with parallel_backend(backend, n_jobs=n_jobs):
#         results = Parallel(verbose=verbose)(tasks)

#     qz_out, R_sigma, R_pi = zip(*results)
#     return np.array(qz_out), np.array(R_sigma), np.array(R_pi)




# def energy_scan(stack: Structure, E_eVs: list[float], theta_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Perform an energy scan at fixed scattering angle.
    
#     Calculates X-ray reflectivity as a function of energy at a fixed scattering
#     angle (theta). Useful for resonant reflectivity measurements and absorption
#     edge studies.
    
#     Args:
#         stack (Structure): The multilayer structure.
#         E_eVs (list[float]): List of X-ray energies in electron volts.
#         theta_deg (float): Scattering angle in degrees.
        
#     Returns:
#         tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
#             - E_np: Energy array in eV
#             - R_sigma_all: Reflectivity for sigma polarization vs energy
#             - R_pi_all: Reflectivity for pi polarization vs energy
#     """
#     structures = []
#     for E_eV in E_eVs:
#         S = _to_pr_structure(stack, E_eV)
#         structures.append(S)
    
#     h = 4.135667696e-15  # Plank's constant eV*s
#     c = 2.99792458e8  # speed of light m/s

#     E_np = np.array(E_eVs)
#     wavelengths = h * c / (E_np * 1e-10)

#     R_sigma_all = []
#     R_pi_all = []
#     for i, E_eV in enumerate(E_eVs):
#         R_sigma, R_pi = pr.Reflectivity(structures[i], theta_deg, wavelengths[i], MagneticCutoff=1e-20)
#         R_sigma_all.append(R_sigma)
#         R_pi_all.append(R_pi)

#     R_sigma_all = np.array(R_sigma_all)
#     R_pi_all = np.array(R_pi_all)
#     return E_np, R_sigma_all, R_pi_all



# def _energy_scan_process(stack, E_eV, theta_deg):
#     """Worker function for multiprocessing energy scan calculations.
    
#     Generates its own structure and calculates reflectivity for a single energy value.
#     Used in parallel processing to avoid shared memory issues.
    
#     Args:
#         stack: The multilayer structure.
#         E_eV (float): X-ray energy in electron volts.
#         theta_deg (float): Scattering angle in degrees.
        
#     Returns:
#         tuple: (R_sigma, R_pi) for the given energy.
#     """
#     structure = _to_pr_structure(stack, E_eV)
#     wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
#     R_sigma, R_pi = pr.Reflectivity(
#         structure,
#         theta_deg,
#         wavelength,
#         MagneticCutoff=1e-20
#     )
#     return R_sigma, R_pi

# def _energy_scan_thread(structure, wavelength, theta_deg):
#     """Worker function for threading energy scan calculations.
    
#     Calculates reflectivity for a single energy using a pre-generated structure.
#     Used in thread-based parallel processing where structures can be shared.
    
#     Args:
#         structure: Pre-generated Pythonreflectivity structure.
#         wavelength (float): X-ray wavelength in Angstroms.
#         theta_deg (float): Scattering angle in degrees.
        
#     Returns:
#         tuple: (R_sigma, R_pi) for the given energy.
#     """
#     R_sigma, R_pi = pr.Reflectivity(
#         structure,
#         theta_deg,
#         wavelength,
#         MagneticCutoff=1e-20
#     )
#     return R_sigma, R_pi

# def energy_scan_parallel(stack: Structure, E_eVs: list[float], theta_deg: float, n_jobs: int = -1, use_threads: bool = False, verbose: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Perform an energy scan using parallel processing.
    
#     Calculates X-ray reflectivity as a function of energy in parallel across
#     multiple energy values using either multiprocessing or threading for 
#     improved performance on large datasets.
    
#     Args:
#         stack (Structure): The multilayer structure.
#         E_eVs (list[float]): List of X-ray energies in electron volts.
#         theta_deg (float): Scattering angle in degrees.
#         n_jobs (int, optional): Number of parallel jobs. -1 uses all cores. Defaults to -1.
#         use_threads (bool, optional): Use threading instead of multiprocessing. Defaults to False.
#         verbose (int, optional): Verbosity level for parallel execution. Defaults to 5.
        
#     Returns:
#         tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
#             - E_np: Energy array in eV
#             - R_sigma_all: Reflectivity for sigma polarization vs energy
#             - R_pi_all: Reflectivity for pi polarization vs energy
#     """
#     E_np = np.array(E_eVs)
#     wavelengths = H_CONST * C_CONST / (E_np * 1e-10)

#     if use_threads:
#         structures = [_to_pr_structure(stack, E) for E in E_eVs]
#         backend = 'threading'
#         tasks = [
#             delayed(_energy_scan_thread)(structures[i], wavelengths[i], theta_deg) for i in range(len(E_eVs))
#         ]
#     else:
#         backend = 'loky'
#         tasks = [
#             delayed(_energy_scan_process)(stack, E, theta_deg) for E in E_eVs
#         ]

#     with parallel_backend(backend, n_jobs=n_jobs):
#         results = Parallel(verbose=verbose)(tasks)

#     R_sigma_all, R_pi_all = zip(*results)
#     return E_np, np.array(R_sigma_all), np.array(R_pi_all)
