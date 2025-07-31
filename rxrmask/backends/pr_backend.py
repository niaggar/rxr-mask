"""Python Reflectivity backend for X-ray reflectometry calculations.

Provides integration with the Python Reflectivity library for
numerical simulation of multilayer structures.
"""

from rxrmask.utils import compute_adaptive_layer_segmentation
from rxrmask.core.compound import Compound
from rxrmask.core.structure import Structure
from rxrmask.core.layer import Layer

import numpy as np
import Pythonreflectivity as pr
from joblib import Parallel, delayed, parallel_backend


# Physical constants
H_CONST = 4.135667696e-15  # Planck constant [eV·s]
C_CONST = 2.99792458e8  # Speed of light [m/s]
QZ_SCALE = 0.001013546247  # Conversion factor qz→theta
HC_EV_ANGSTROM = 12398.41984  # hc constant [eV·Å]




def _print_structure_info(structure):
    """Print debug information about layer structure."""
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

def _compound_lookup_table(stack: Structure) -> list[tuple[float, Compound]]:
    """Build sorted lookup table mapping depth to compounds."""
    z = 0.0
    table = []
    for compound in stack.compounds:
        if compound is not None:
            z += compound.thickness.get()
            table.append((z, compound))
    return table

def _find_compound_at_z(z: float, table: list[tuple[float, Compound]]) -> Compound:
    """Find compound at given depth position."""
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
    """Build PythonReflectivity structure from layer segments.
    
    Each segment is represented by first layer but with summed thickness.
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
    """Calculate X-ray reflectivity for given momentum transfer and energy.
    
    Args:
        stack: Multilayer structure.
        qz: Momentum transfer array.
        E_eV: X-ray energy in eV.
        precision: Adaptive layer segmentation precision.
        
    Returns:
        tuple: (qz, R_sigma, R_pi) reflectivity data.
    """
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

def _worker_process(stack: Structure, qz_i: float, E_eV: float, precision: float = 1e-6):
    """Worker function for parallel reflectivity calculation."""
    for layer in stack.layers:
        layer.compute(E_eV)
    compound_map = _compound_lookup_table(stack)
    indices = compute_adaptive_layer_segmentation(stack.layers, E_eV, precision)
    structure = _to_pr_structure_from_segments(stack.layers, indices, E_eV, compound_map)

    wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
    theta = np.arcsin(qz_i / E_eV / QZ_SCALE) * 180 / np.pi
    R_sigma, R_pi = pr.Reflectivity(structure, theta, wavelength, MagneticCutoff=1e-20)
    return qz_i, R_sigma, R_pi

def _worker_thread(structure, qz_i: float, E_eV: float):
    """Worker function for threaded reflectivity calculation."""
    wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
    theta = np.arcsin(qz_i / E_eV / QZ_SCALE) * 180 / np.pi
    R_sigma, R_pi = pr.Reflectivity(structure, theta, wavelength, MagneticCutoff=1e-20)
    return qz_i, R_sigma, R_pi

def reflectivity_parallel(stack: Structure, qz: np.ndarray, E_eV: float, precision: float = 1e-6, n_jobs: int = -1, use_threads: bool = False, verbose: int = 5):
    """Calculate reflectivity in parallel for multiple qz values.
    
    Args:
        stack: Multilayer structure.
        qz: Momentum transfer array.
        E_eV: X-ray energy in eV.
        precision: Adaptive segmentation precision.
        n_jobs: Number of parallel jobs.
        use_threads: Use threading instead of processes.
        verbose: Verbosity level.
        
    Returns:
        tuple: (qz, R_sigma, R_pi) reflectivity arrays.
    """
    if use_threads:
        for layer in stack.layers:
            layer.compute(E_eV)
        compound_map = _compound_lookup_table(stack)
        indices = compute_adaptive_layer_segmentation(stack.layers, E_eV, precision)
        structure = _to_pr_structure_from_segments(stack.layers, indices, E_eV, compound_map)

        tasks = [delayed(_worker_thread)(structure, qzi, E_eV) for qzi in qz]
        backend = 'threading'
    else:
        tasks = [delayed(_worker_process)(stack, qzi, E_eV, precision) for qzi in qz]
        backend = 'loky'

    with parallel_backend(backend, n_jobs=n_jobs):
        results = Parallel(verbose=verbose)(tasks)

    qz_out, R_sigma, R_pi = zip(*results)
    return np.array(qz_out), np.array(R_sigma), np.array(R_pi)





def energy_scan(stack: Structure, E_eVs: list[float], theta_deg: float, precision: float = 1e-6):
    """Energy scan at fixed scattering angle using adaptive segmentation.

    Args:
        stack: Multilayer structure.
        E_eVs: X-ray energies in eV.
        theta_deg: Fixed incidence angle in degrees.
        precision: Adaptive segmentation precision.

    Returns:
        tuple: (E array, R_sigma array, R_pi array).
    """
    import time
    start_time = time.time()
    
    R_sigma_all = []
    R_pi_all = []
    compound_map = _compound_lookup_table(stack)
    
    comp_time = time.time() - start_time
    print(f"Compound map creation took {comp_time:.5f} seconds")
    
    wavelength_energies = H_CONST * C_CONST / (np.array(E_eVs) * 1e-10)
    structures_energies = []
    
    wave_time = time.time() - start_time
    print(f"Wavelength calculation took {wave_time:.5f} seconds")

    for E in E_eVs:
        for layer in stack.layers:
            layer.compute(E)
            
        indices = compute_adaptive_layer_segmentation(stack.layers, E, precision)
        structure = _to_pr_structure_from_segments(stack.layers, indices, E, compound_map)
        structures_energies.append(structure)

    structure_time = time.time() - start_time
    print(f"Structure creation took {structure_time:.5f} seconds")
    
    for i, E_eV in enumerate(E_eVs):
        wavelength = wavelength_energies[i]
        structure = structures_energies[i]
        
        R_sigma, R_pi = pr.Reflectivity(structure, theta_deg, wavelength, MagneticCutoff=1e-20)
        R_sigma_all.append(R_sigma)
        R_pi_all.append(R_pi)
        
    calculation_time = time.time() - start_time
    print(f"Reflectivity calculation took {calculation_time:.5f} seconds")
        
    return np.array(E_eVs), np.array(R_sigma_all), np.array(R_pi_all)

def _energy_scan_worker(stack: Structure, E_eV: float, theta_deg: float, precision: float = 1e-6):
    """Worker function for parallel energy scan."""
    for layer in stack.layers:
        layer.compute(E_eV)
    compound_map = _compound_lookup_table(stack)
    indices = compute_adaptive_layer_segmentation(stack.layers, E_eV, precision)
    structure = _to_pr_structure_from_segments(stack.layers, indices, E_eV, compound_map)

    wavelength = H_CONST * C_CONST / (E_eV * 1e-10)
    R_sigma, R_pi = pr.Reflectivity(structure, theta_deg, wavelength, MagneticCutoff=1e-20)
    return R_sigma, R_pi

def _energy_scan_thread(structure, wavelength: float, theta_deg: float):
    """Worker function for threaded energy scan."""
    R_sigma, R_pi = pr.Reflectivity(structure, theta_deg, wavelength, MagneticCutoff=1e-20)
    return R_sigma, R_pi

def energy_scan_parallel(stack: Structure, E_eVs: list[float], theta_deg: float, precision: float = 1e-6, n_jobs: int = -1, use_threads: bool = False, verbose: int = 5):
    """Parallel energy scan at fixed scattering angle.
    
    Args:
        stack: Multilayer structure.
        E_eVs: X-ray energies in eV.
        theta_deg: Fixed incidence angle in degrees.
        precision: Adaptive segmentation precision.
        n_jobs: Number of parallel jobs.
        use_threads: Use threading instead of processes.
        verbose: Verbosity level.
        
    Returns:
        tuple: (E array, R_sigma array, R_pi array).
    """
    E_np = np.array(E_eVs)
    wavelengths = H_CONST * C_CONST / (E_np * 1e-10)

    if use_threads:
        structures = []
        for E in E_eVs:
            for layer in stack.layers:
                layer.compute(E)
            compound_map = _compound_lookup_table(stack)
            indices = compute_adaptive_layer_segmentation(stack.layers, E, precision)
            S = _to_pr_structure_from_segments(stack.layers, indices, E, compound_map)
            structures.append(S)

        tasks = [delayed(_energy_scan_thread)(structures[i], wavelengths[i], theta_deg) for i in range(len(E_np))]
        backend = 'threading'
    else:
        tasks = [delayed(_energy_scan_worker)(stack, E, theta_deg, precision) for E in E_eVs]
        backend = 'loky'

    with parallel_backend(backend, n_jobs=n_jobs):
        results = Parallel(verbose=verbose)(tasks)

    R_sigma_all, R_pi_all = zip(*results)
    return E_np, np.array(R_sigma_all), np.array(R_pi_all)
