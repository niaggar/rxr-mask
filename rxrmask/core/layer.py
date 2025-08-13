from rxrmask.core.atom import Atom
from rxrmask.core.parameter import Parameter

from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt


h = 4.135667696e-15  # Planck's constant [eV s]
c = 2.99792450e10  # Speed of light [cm/s]
re = 2.817940322719e-13  # Classical electron radius [cm]
avocado = 6.02214076e23  # Avogadro's number [mol^-1]
pihc = 2 * np.pi / (h * c)  # Photon wavenumber factor [1/cm]
pireav = 2 * np.pi * re * avocado  # Electron density factor [cm^-3]


@dataclass
class AtomLayer:
    """Atomic species in a layer with density and form factors.

    Attributes:
        atom: Atom object with properties and form factors
        molar_density: Molar density in mol/cm³
        molar_magnetic_density: Magnetic density in mol/cm³
    """

    atom: Atom
    z_deepness: np.ndarray
    molar_density: np.ndarray  # in mol/cm^3
    molar_magnetic_density: np.ndarray  # in mol/cm^3

    def get_f1f2(self, energy_eV: float, *kwargs) -> tuple[float, float]:
        """Get atomic form factors f1 and f2 at specific energy.

        Args:
            energy_eV: X-ray energy in eV

        Returns:
            (f1, f2) form factor components
        """
        return self.atom.ff.get_formfactors(energy_eV, *kwargs)

    def get_q1q2(self, energy_eV: float, *kwargs) -> tuple[float, float]:
        """Get magnetic form factors q1 and q2 at specific energy.

        Args:
            energy_eV: X-ray energy in eV

        Returns:
            (q1, q2) magnetic form factors, (0, 0) if non-magnetic.
        """
        if self.atom.ffm is None:
            return 0.0, 0.0
        return self.atom.ffm.get_formfactors(energy_eV, *kwargs)

    def get_f1f2_energies(self, energies: npt.NDArray[np.float64], *kwargs) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get atomic form factors for multiple energies.

        Args:
            energies: Array of energies in eV.

        Returns:
            (f1, f2) form factor arrays.
        """
        return self.atom.ff.get_formfactors_energies(energies, *kwargs)

    def get_q1q2_energies(self, energies: npt.NDArray[np.float64], *kwargs) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get magnetic form factors for multiple energies.

        Args:
            energies: Array of energies in eV.

        Returns:
            (q1, q2) magnetic form factor arrays, zeros if non-magnetic.
        """
        if self.atom.ffm is None:
            return (np.zeros(len(energies)), np.zeros(len(energies)))
        return self.atom.ffm.get_formfactors_energies(energies, *kwargs)


def get_index_of_refraction(layers: list[AtomLayer], energy_eV: float, *kwargs) -> npt.NDArray[np.complexfloating]:
    k0 = energy_eV * pihc
    constant = pireav / (k0**2)
    f1_layer = np.zeros(len(layers[0].z_deepness), dtype=np.float64)
    f2_layer = np.zeros(len(layers[0].z_deepness), dtype=np.float64)

    for layer in layers:
        if not np.any(layer.molar_density > 0):
            continue

        f1, f2 = layer.get_f1f2(energy_eV, *kwargs)

        f1_layer += f1 * layer.molar_density
        f2_layer += f2 * layer.molar_density

    delta = constant * f1_layer
    beta = constant * f2_layer
    n_complex = 1 - delta + 1j * beta

    return n_complex


def get_magnetic_optical_constant(layers: list[AtomLayer], energy_eV: float, *kwargs) -> npt.NDArray[np.complexfloating]:
    k0 = energy_eV * pihc
    constant = pireav / (k0**2)
    q1_layer = np.zeros(len(layers[0].z_deepness), dtype=np.float64)
    q2_layer = np.zeros(len(layers[0].z_deepness), dtype=np.float64)

    for layer in layers:
        if not np.any(layer.molar_magnetic_density > 0):
            continue

        q1, q2 = layer.get_q1q2(energy_eV, *kwargs)

        q1_layer += q1 * layer.molar_magnetic_density
        q2_layer += q2 * layer.molar_magnetic_density

    delta_m = constant * q1_layer
    beta_m = constant * q2_layer

    q_complex = 2 * (delta_m + 1j * beta_m)
    return q_complex


def get_index_of_refraction_batch(layers: list[AtomLayer], energies: npt.NDArray[np.float64], *kwargs) -> npt.NDArray[np.complexfloating]:
    m = len(energies)
    k0 = energies * pihc  # (m,)
    constant = pireav / (k0**2)  # (m,)
    const_col = constant[:, None]  # (m, 1)

    n_slices = len(layers[0].z_deepness)
    f1_tot = np.zeros((m, n_slices), dtype=np.float64)
    f2_tot = np.zeros((m, n_slices), dtype=np.float64)

    for layer in layers:
        rho = layer.molar_density
        if not np.any(rho > 0):
            continue

        f1_s, f2_s = layer.get_f1f2_energies(energies, *kwargs)
        f1_s = np.asarray(f1_s, dtype=np.float64).reshape(m, 1)  # (m,1)
        f2_s = np.asarray(f2_s, dtype=np.float64).reshape(m, 1)  # (m,1)
        f1_tot += const_col * f1_s * rho[np.newaxis, :]
        f2_tot += const_col * f2_s * rho[np.newaxis, :]

    n_complex = (1.0 - f1_tot) + 1j * f2_tot  # (m, n)
    return n_complex


def get_magnetic_optical_constant_batch(layers: list[AtomLayer], energies: npt.NDArray[np.float64], *kwargs) -> npt.NDArray[np.complexfloating]:
    m = len(energies)
    k0 = energies * pihc  # (m,)
    constant = pireav / (k0**2)  # (m,)
    const_col = constant[:, None]  # (m, 1)

    n_slices = len(layers[0].z_deepness)
    q1_tot = np.zeros((m, n_slices), dtype=np.float64)
    q2_tot = np.zeros((m, n_slices), dtype=np.float64)

    for layer in layers:
        rho_m = layer.molar_magnetic_density
        if not np.any(rho_m > 0):
            continue

        q1_s, q2_s = layer.get_q1q2_energies(energies, *kwargs)
        q1_s = np.asarray(q1_s, dtype=np.float64).reshape(m, 1)  # (m,1)
        q2_s = np.asarray(q2_s, dtype=np.float64).reshape(m, 1)  # (m,1)
        q1_tot += const_col * q1_s * rho_m[np.newaxis, :]
        q2_tot += const_col * q2_s * rho_m[np.newaxis, :]

    q_complex = 2 * (q1_tot + 1j * q2_tot)  # (m, n)
    return q_complex
