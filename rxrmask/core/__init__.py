"""Core X-ray reflectometry calculation modules."""

from .atom import Atom, find_atom
from .formfactor import FormFactorModel, FormFactorLocalDB, FormFactorVacancy, FormFactorFile
from .compound import CompoundDetails, Compound, create_compound
from .layer import (
    AtomLayer,
    get_index_of_refraction,
    get_index_of_refraction_batch,
    get_magnetic_optical_constant,
    get_magnetic_optical_constant_batch,
)
from .structure import Structure
from .parameter import Parameter, DerivedParameter, ParametersContainer, DependentParameter
from .reflectivitydata import SimReflectivityData, SimEnergyScanData, EnergyScan, ReflectivityScan
from .loaddata import load_reflectivity_scans, load_energy_scans

__all__ = [
    "Atom",
    "find_atom",
    "FormFactorModel",
    "FormFactorLocalDB",
    "FormFactorVacancy",
    "FormFactorFile",
    "CompoundDetails",
    "Compound",
    "create_compound",
    "AtomLayer",
    "get_index_of_refraction",
    "get_index_of_refraction_batch",
    "get_magnetic_optical_constant",
    "get_magnetic_optical_constant_batch",
    "Structure",
    "Parameter",
    "DependentParameter",
    "DerivedParameter",
    "ParametersContainer",
    "SimReflectivityData",
    "SimEnergyScanData",
    "EnergyScan",
    "ReflectivityScan",
    "load_reflectivity_scans",
    "load_energy_scans",
]
