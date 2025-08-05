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
HC_WAVELENGTH_CONV = H_CONST * C_CONST * 1e10  # Pre-calculated h*c conversion for wavelength (eV·Å)


def _print_structure_info(structure):
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
        print(
            f"Layer {i}: d={d:.2f} Å, sigma={sigma}, eps={eps}, mag={mag}, epsg={epsg}"
        )


def _compound_lookup_table(stack: Structure) -> list[tuple[float, Compound]]:
    z = 0.0
    table = []
    for compound in stack.compounds:
        if compound is not None:
            z += compound.thickness.get()
            table.append((z, compound))
    return table


def _find_compound_at_z(z: float, table: list[tuple[float, Compound]]) -> Compound:
    for z_top, comp in table:
        if z <= z_top:
            return comp
    return table[-1][1]


def _to_pr_structure_from_segments(
    eps: np.ndarray,
    eps_mag: np.ndarray,
    thicknesses: np.ndarray,
    indices: list[int],
    compound_map: list[tuple[float, Compound]],
):
    n_slabs = len(indices)
    structure = pr.Generate_structure(n_slabs)
    z = 0.0

    idx = 0
    j = 0
    for i in indices:
        thickness_segment = sum(thicknesses[j : i + 1])
        eps_segment = eps[j]
        eps_mag_segment = eps_mag[j]

        z += thickness_segment
        compound = _find_compound_at_z(z, compound_map)
        structure[idx].setmag(compound.magnetic_direction)
        structure[idx].seteps([eps_segment, eps_segment, eps_segment, eps_mag_segment])
        structure[idx].setd(thickness_segment)

        j = i + 1
        idx += 1

    return structure


def reflectivity(
    stack: Structure,
    qz: np.ndarray,
    E_eV: float,
    precision: float = 1e-6,
    als: bool = True,
):
    if stack.n_layers == 0:
        raise ValueError(
            "Stack has no layers. Please create layers before computing reflectivity."
        )

    compound_map = _compound_lookup_table(stack)

    index_of_refraction = np.array(
        [layer.get_index_of_refraction(E_eV) for layer in stack.layers]
    )
    magnetic_optical_constants = np.array(
        [layer.get_magnetic_optical_constant(E_eV) for layer in stack.layers]
    )
    thicknesses = np.array([layer.thickness for layer in stack.layers])

    eps = index_of_refraction**2
    eps_mag = -1 * eps * magnetic_optical_constants

    indices = compute_adaptive_layer_segmentation(
        index_of_refraction, magnetic_optical_constants, precision, als=als
    )
    structure = _to_pr_structure_from_segments(
        eps, eps_mag, thicknesses, indices, compound_map
    )

    wavelength = HC_WAVELENGTH_CONV / E_eV  # wavelength of incoming x-ray

    theta_deg = np.arcsin(qz / E_eV / QZ_SCALE) * 180 / np.pi
    R_sigma, R_pi = pr.Reflectivity(
        structure, theta_deg, wavelength, MagneticCutoff=1e-20
    )

    return qz, R_sigma, R_pi


def _worker_thread(structure, qz_i: float, E_eV: float):
    wavelength = HC_WAVELENGTH_CONV / E_eV
    theta = np.arcsin(qz_i / E_eV / QZ_SCALE) * 180 / np.pi
    R_sigma, R_pi = pr.Reflectivity(structure, theta, wavelength, MagneticCutoff=1e-20)
    return qz_i, R_sigma, R_pi


def reflectivity_parallel(
    stack: Structure,
    qz: np.ndarray,
    E_eV: float,
    precision: float = 1e-6,
    n_jobs: int = -1,
    use_threads: bool = False,
    verbose: int = 0,
    als: bool = True,
):
    if stack.n_layers == 0:
        raise ValueError(
            "Stack has no layers. Please create layers before computing reflectivity."
        )

    if use_threads:
        compound_map = _compound_lookup_table(stack)
        index_of_refraction = np.array(
            [layer.get_index_of_refraction(E_eV) for layer in stack.layers]
        )
        magnetic_optical_constants = np.array(
            [layer.get_magnetic_optical_constant(E_eV) for layer in stack.layers]
        )
        thicknesses = np.array([layer.thickness for layer in stack.layers])

        eps = index_of_refraction**2
        eps_mag = -1 * eps * magnetic_optical_constants
        indices = compute_adaptive_layer_segmentation(
            index_of_refraction,
            magnetic_optical_constants,
            precision,
            als=als,
        )
        structure = _to_pr_structure_from_segments(
            eps, eps_mag, thicknesses, indices, compound_map
        )

        tasks = [delayed(_worker_thread)(structure, qzi, E_eV) for qzi in qz]
        backend = "threading"
    else:
        tasks = [delayed(reflectivity)(stack, qzi, E_eV, precision) for qzi in qz]
        backend = "loky"

    with parallel_backend(backend, n_jobs=n_jobs):
        results = Parallel(verbose=verbose)(tasks)

    qz_out, R_sigma, R_pi = zip(*results)
    return np.array(qz_out), np.array(R_sigma), np.array(R_pi)


def get_index_of_refraction_energies(stack: Structure, energies: np.ndarray):
    index_energies = np.zeros((len(energies), len(stack.layers)), dtype=np.complex128)

    for i, layer in enumerate(stack.layers):
        index_layer = layer.get_index_of_refraction_batch(energies)
        index_energies[:, i] = index_layer

    return index_energies


def get_magnetic_optical_constants_energies(stack: Structure, energies: np.ndarray):
    mag_constants = np.zeros((len(energies), len(stack.layers)), dtype=np.complex128)

    for i, layer in enumerate(stack.layers):
        mag_layer = layer.get_magnetic_optical_constant_batch(energies)
        mag_constants[:, i] = mag_layer

    return mag_constants


def energy_scan(
    stack: Structure,
    E_eVs: list[float],
    theta_deg: float,
    precision: float = 1e-6,
    als: bool = True,
):
    if stack.n_layers == 0:
        raise ValueError(
            "Stack has no layers. Please create layers before computing reflectivity."
        )

    compound_map = _compound_lookup_table(stack)

    e_array = np.array(E_eVs)
    wavelength_energies = HC_WAVELENGTH_CONV / e_array
    structures_energies = []

    index_of_refraction_energies = get_index_of_refraction_energies(stack, e_array)
    magnetic_optical_constants_energies = get_magnetic_optical_constants_energies(
        stack, e_array
    )
    thicknesses = np.array([layer.thickness for layer in stack.layers])

    eps = np.array(
        [index_of_refraction_energies[i, :] ** 2 for i in range(len(e_array))]
    )
    eps_mag = np.array(
        [
            -1 * eps[i] * magnetic_optical_constants_energies[i, :]
            for i in range(len(e_array))
        ]
    )

    indices_energies = [
        compute_adaptive_layer_segmentation(
            index_of_refraction_energies[i, :],
            magnetic_optical_constants_energies[i, :],
            precision,
            als=als,
        )
        for i in range(len(e_array))
    ]
    structures_energies = [
        _to_pr_structure_from_segments(
            eps[i, :], eps_mag[i, :], thicknesses, indices_energies[i], compound_map
        )
        for i in range(len(e_array))
    ]

    R_sigma_all = []
    R_pi_all = []
    for i, structure in enumerate(structures_energies):
        wavelength = wavelength_energies[i]
        R_sigma, R_pi = pr.Reflectivity(
            structure, theta_deg, wavelength, MagneticCutoff=1e-20
        )
        R_sigma_all.append(R_sigma)
        R_pi_all.append(R_pi)

    return e_array, R_sigma_all, R_pi_all


def _energy_scan_thread(structure, wavelength, theta_deg):
    R_sigma, R_pi = pr.Reflectivity(structure, theta_deg, wavelength, MagneticCutoff=1e-20)
    return R_sigma, R_pi

def _energy_scan_worker(stack, E_eV, theta_deg, precision, als):
    compound_map = _compound_lookup_table(stack)

    index_of_refraction = np.array(
        [layer.get_index_of_refraction(E_eV) for layer in stack.layers]
    )
    magnetic_optical_constants = np.array(
        [layer.get_magnetic_optical_constant(E_eV) for layer in stack.layers]
    )
    thicknesses = np.array([layer.thickness for layer in stack.layers])

    eps = index_of_refraction**2
    eps_mag = -1 * eps * magnetic_optical_constants

    indices = compute_adaptive_layer_segmentation(
        index_of_refraction, magnetic_optical_constants, precision, als=als
    )
    structure = _to_pr_structure_from_segments(
        eps, eps_mag, thicknesses, indices, compound_map
    )

    wavelength = HC_WAVELENGTH_CONV / E_eV  # wavelength of incoming x-ray
    R_sigma, R_pi = pr.Reflectivity(
        structure, theta_deg, wavelength, MagneticCutoff=1e-20
    )

    return R_sigma, R_pi

def energy_scan_parallel(
    stack, E_eVs, theta_deg, precision=1e-6, n_jobs=-1, verbose=0, als=True, use_threads=False
):
    e_array = np.array(E_eVs)

    if use_threads:
        compound_map = _compound_lookup_table(stack)
        index_energies = get_index_of_refraction_energies(stack, e_array)
        mag_constants = get_magnetic_optical_constants_energies(stack, e_array)
        thicknesses = np.array([layer.thickness for layer in stack.layers])

        eps = index_energies**2
        eps_mag = -1 * eps * mag_constants

        indices_energies = [
            compute_adaptive_layer_segmentation(index_energies[i], mag_constants[i], precision, als=als)
            for i in range(len(e_array))
        ]

        structures = [
            _to_pr_structure_from_segments(
                eps[i], eps_mag[i], thicknesses, indices_energies[i], compound_map
            )
            for i in range(len(e_array))
        ]

        wavelengths = HC_WAVELENGTH_CONV / e_array

        with parallel_backend("threading", n_jobs=n_jobs):
            results = Parallel(verbose=verbose)(
                delayed(_energy_scan_thread)(structures[i], wavelengths[i], theta_deg)
                for i in range(len(e_array))
            )

    else:
        with parallel_backend("loky", n_jobs=n_jobs):
            results = Parallel(verbose=verbose)(
                delayed(_energy_scan_worker)(stack, E, theta_deg, precision, als)
                for E in e_array
            )

    R_sigma_all, R_pi_all = zip(*results)
    return e_array, list(R_sigma_all), list(R_pi_all)
