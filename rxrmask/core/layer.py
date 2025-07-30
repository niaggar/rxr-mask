"""Layer module for RXR-Mask.

This module provides classes for representing layers in multilayer structures
used in X-ray reflectometry calculations. 
"""

from rxrmask.core.atom import Atom
from rxrmask.core.parameter import Parameter

from dataclasses import dataclass, field
import numpy as np


# Physical constants
h = 4.135667696e-15  # Planck's Constant [eV s]
c = 2.99792450e10  # Speed of light in vacuum [cm/s]
re = 2.817940322719e-13  # Classical electron radius (Thompson scattering length) [cm]
avocado = 6.02214076e23  # Avogadro's number [mol^-1]
pihc = 2 * np.pi / (h * c)  # To calculate the photon wavenumber in vacuum [1/cm]
pireav = 2 * np.pi * re * avocado  # To calculate the electron density in cm^-3


@dataclass
class AtomLayer:
    """Represents a specific atomic species in a layer.
    
    Attributes:
        atom (Atom): The Atom object containing atomic properties and form factors.
        molar_density (float): Molar density of the element in mol/cm³.
        molar_magnetic_density (float | None): Molar magnetic density in mol/cm³. 
                                              Defaults to None.
    """
    atom: Atom
    molar_density: Parameter[float]  # in mol/cm^3
    molar_magnetic_density: Parameter[float] | None = None  # in mol/cm^3

    def get_f1f2(self, energy_eV: float, *args) -> tuple[float, float]:
        """Get atomic form factors f1 and f2 at a specific energy.
        
        Args:
            energy_eV (float): X-ray energy in electron volts.
            
        Returns:
            tuple[float, float]: Tuple containing (f1, f2) form factor components.
        """
        return self.atom.ff.get_formfactors(energy_eV, *args)

    def get_q1q2(self, energy_eV: float, *args) -> tuple[float, float]:
        """Get magnetic form factors q1 and q2 at a specific energy.
        
        Args:
            energy_eV (float): X-ray energy in electron volts.
            
        Returns:
            tuple[float, float]: Tuple containing (q1, q2) magnetic form factor components.
                                Returns (0.0, 0.0) if atom has no magnetic form factor.
        """
        if self.atom.ffm is None:
            return 0.0, 0.0
        return self.atom.ffm.get_formfactors(energy_eV, *args)


@dataclass
class Layer:
    """Represents a single layer in a multilayer structure.
    
    This class encapsulates the properties of a layer including its thickness
    and atomic composition. It provides methods to calculate optical constants
    based on the constituent elements and their densities.
    
    Attributes:
        id (str): Unique identifier for the layer.
        thickness (float): Layer thickness in Angstroms.
        elements (list[ElementLayer]): List of atomic elements in the layer.
    """
    id: str
    thickness: float  # in Angstroms
    elements: list[AtomLayer] = field(default_factory=list)

    def get_index_of_refraction(self, energy_eV: float, *args) -> complex:
        """Calculate the complex refractive index of the layer.
        
        Args:
            energy_eV (float): X-ray energy in electron volts.
            
        Returns:
            complex: Complex refractive index n = 1 - δ + iβ where:
                    - δ (delta) is the real part of the refractive index decrement
                    - β (beta) is the imaginary part related to absorption
        """
        f1_layer = 0.0
        f2_layer = 0.0
        for l in self.elements:
            f1, f2 = l.get_f1f2(energy_eV, *args)
            molar_density = l.molar_density.get()
            
            f1_layer += f1 * molar_density
            f2_layer += f2 * molar_density

        k0 = energy_eV * pihc
        constant = pireav / (k0 ** 2)
        
        delta = constant * f1_layer
        beta  = constant * f2_layer
        n_complex = 1 - delta + 1j * beta

        return n_complex

    def get_magnetic_optical_constant(self, energy_eV: float, *args) -> complex:
        """Calculate the magnetic optical constant of the layer.
        
        Args:
            energy_eV (float): X-ray energy in electron volts.
            
        Returns:
            complex: Magnetic optical constant Q = 2(q1 + iq2) where:
                    - q1 is the real part of the magnetic optical constant
                    - q2 is the imaginary part related to magnetic absorption
        """
        q1_layer = 0.0
        q2_layer = 0.0
        for l in self.elements:
            q1, q2 = l.get_q1q2(energy_eV, *args)
            molar_magnetic_density = l.molar_magnetic_density.get() if l.molar_magnetic_density else 0.0
            
            q1_layer += q1 * molar_magnetic_density
            q2_layer += q2 * molar_magnetic_density

        k0 = energy_eV * pihc
        constant = pireav / (k0 ** 2)

        delta_m = constant * q1_layer
        beta_m  = constant * q2_layer

        Q = 2 * (delta_m + 1j * beta_m)
        return Q
    