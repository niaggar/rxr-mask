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
    molar_density: Parameter[float]  # in mol/cm^3
    molar_magnetic_density: Parameter[float] | None = None  # in mol/cm^3

    def get_f1f2(self, energy_eV: float, *args) -> tuple[float, float]:
        """Get atomic form factors f1 and f2 at specific energy.
        
        Args:
            energy_eV: X-ray energy in eV
            
        Returns:
            (f1, f2) form factor components
        """
        return self.atom.ff.get_formfactors(energy_eV, *args)

    def get_q1q2(self, energy_eV: float, *args) -> tuple[float, float]:
        """Get magnetic form factors q1 and q2 at specific energy.
        
        Args:
            energy_eV: X-ray energy in eV
            
        Returns:
            (q1, q2) magnetic form factors, (0, 0) if non-magnetic.
        """
        if self.atom.ffm is None:
            return 0.0, 0.0
        return self.atom.ffm.get_formfactors(energy_eV, *args)
    
    def get_f1f2_energies(self, energies: npt.NDArray[np.float64], *args) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get atomic form factors for multiple energies.

        Args:
            energies: Array of energies in eV.

        Returns:
            (f1, f2) form factor arrays.
        """
        return self.atom.ff.get_formfactors_energies(energies, *args)

    def get_q1q2_energies(self, energies: npt.NDArray[np.float64], *args) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get magnetic form factors for multiple energies.

        Args:
            energies: Array of energies in eV.

        Returns:
            (q1, q2) magnetic form factor arrays, zeros if non-magnetic.
        """
        if self.atom.ffm is None:
            return (np.zeros(len(energies)), np.zeros(len(energies)))
        return self.atom.ffm.get_formfactors_energies(energies, *args)


@dataclass
class Layer:
    """Layer in multilayer structure with atomic composition.
    
    Attributes:
        id: Layer identifier
        thickness: Layer thickness in Angstroms
        elements: List of atomic species in the layer
    """
    id: str
    thickness: float  # in Angstroms
    elements: list[AtomLayer] = field(default_factory=list)


    def get_index_of_refraction(self, energy_eV: float, *args) -> complex:
        """Get complex refractive index for given energy."""
        f1_layer = 0.0
        f2_layer = 0.0
        for l in self.elements:
            molar_density = l.molar_density.get()
            if molar_density == 0.0:
                continue
            f1, f2 = l.get_f1f2(energy_eV, *args)
            
            f1_layer += f1 * molar_density
            f2_layer += f2 * molar_density

        k0 = energy_eV * pihc
        constant = pireav / (k0 ** 2)
        
        delta = constant * f1_layer
        beta  = constant * f2_layer
        n_complex = 1 - delta + 1j * beta
 
        return n_complex
        
    def get_magnetic_optical_constant(self, energy_eV: float, *args) -> complex:
        """Get magnetic optical constant for given energy."""
        q1_layer = 0.0
        q2_layer = 0.0
        for l in self.elements:
            molar_magnetic_density = l.molar_magnetic_density.get() if l.molar_magnetic_density else 0.0
            if molar_magnetic_density == 0.0:
                continue
            q1, q2 = l.get_q1q2(energy_eV, *args)
            
            q1_layer += q1 * molar_magnetic_density
            q2_layer += q2 * molar_magnetic_density

        k0 = energy_eV * pihc
        constant = pireav / (k0 ** 2)

        delta_m = constant * q1_layer
        beta_m  = constant * q2_layer

        q_complex = 2 * (delta_m + 1j * beta_m)
        return q_complex
        
    
    def get_index_of_refraction_batch(self, energies: npt.NDArray[np.float64], *args) -> npt.NDArray[np.complexfloating]:
        """Get complex refractive index for multiple energies."""
        delta = np.zeros(len(energies), dtype=np.float64)
        beta = np.zeros(len(energies), dtype=np.float64)

        k0 = energies * pihc
        constant = pireav / (k0 ** 2)
        
        for l in self.elements:
            molar_density = l.molar_density.get()
            if molar_density == 0.0:
                continue

            formFactors = l.get_f1f2_energies(energies, *args)
            delta += constant * formFactors[0] * molar_density
            beta += constant * formFactors[1] * molar_density

        n_complex = 1 - delta + 1j * beta
        return n_complex
    
    def get_magnetic_optical_constant_batch(self, energies: npt.NDArray[np.float64], *args) -> npt.NDArray[np.complexfloating]:
        """Get magnetic optical constant for multiple energies."""
        delta_m = np.zeros(len(energies), dtype=np.float64)
        beta_m = np.zeros(len(energies), dtype=np.float64)

        k0 = energies * pihc
        constant = pireav / (k0 ** 2)

        for l in self.elements:
            molar_magnetic_density = l.molar_magnetic_density.get() if l.molar_magnetic_density else 0.0
            if molar_magnetic_density == 0.0:
                continue

            formFactors = l.get_q1q2_energies(energies, *args)
            delta_m += constant * formFactors[0] * molar_magnetic_density
            beta_m += constant * formFactors[1] * molar_magnetic_density

        q_complex = 2 * (delta_m + 1j * beta_m)
        return q_complex
    
    def print_details(self):
        print(f"Layer ID: {self.id}")
        print(f"  Thickness: {self.thickness} Angstroms")
        for element in self.elements:
            atom = element.atom
            print(
                f"  Element: {atom.name}, "
                f"Molar Density: {element.molar_density.get()} mol/cm³, "
                f"Magnetic Density: {element.molar_magnetic_density.get() if element.molar_magnetic_density else 'N/A'} mol/cm³, "
                f"Is Magnetic: {'Yes' if element.molar_magnetic_density else 'No'}"
            )
