from rxrmask.core.atom import Atom, find_atom
from rxrmask.core.parameter import Parameter, ParametersContainer

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CompoundDetails:
    """Atomic species within a compound layer.

    Attributes:
        index: Atom position in the lattice layer
        name: Chemical symbol
        stochiometric_fraction: Stoichiometric coefficient
        atom: Atom object with atomic properties
        thickness: Layer thickness in Angstroms
        roughness: Interface roughness in Angstroms
        prev_roughness: Previous interface roughness in Angstroms
        molar_density: Molar density in mol/cm³
        molar_magnetic_density: Magnetic density in mol/cm³
    """

    index: int
    name: str
    stochiometric_fraction: float
    atom: Atom
    thickness: Parameter[float]  # in Angstrom
    roughness: Parameter[float]  # in Angstrom
    prev_roughness: Parameter[float]  # in Angstrom
    molar_density: Parameter[float]  # in mol/cm^3
    molar_magnetic_density: Parameter[float]  # in mol/cm^3

    def __init__(self):
        self.name = ""
        self.stochiometric_fraction = 0.0


@dataclass
class Compound:
    """Chemical compound in multilayer structure.

    Attributes:
        name: Compound name
        formula: Chemical formula as "Element1:count1,Element2:count2"
        thickness: Total thickness in Angstroms
        roughness: Interface roughness in Angstroms
        prev_roughness: Previous interface roughness in Angstroms
        linked_prev_roughness: Whether to link previous roughness
        density: Density in g/cm³
        magnetic_density: Magnetic density in g/cm³
        magnetic: Whether compound is magnetic
        magnetic_direction: Magnetic moment direction
        compound_details: Atomic layer structures
    """

    name: str
    formula: str

    thickness: Parameter[float]  # in Angstrom
    roughness: Parameter[float]  # in Angstrom
    prev_roughness: Parameter[float]  # in Angstrom
    linked_prev_roughness: bool
    density: Parameter[float]  # in g/cm^3
    magnetic_density: Parameter[float]  # in g/cm^3

    magnetic: bool = False
    magnetic_direction: Literal["x", "y", "z", "0"] = "0"
    compound_details: list[CompoundDetails] = field(default_factory=list)

    def __init__(self):
        self.name = ""
        self.formula = ""
        self.compound_details = []
        self.magnetic = False
        self.magnetic_direction = "0"

    def print_details(self):
        print(f"Compound: {self.name}")
        print(f"  Formula: {self.formula}")
        print(f"  Thickness: {self.thickness.get()} Angstrom")
        print(f"  Density: {self.density.get()} g/cm³")
        print(f"  Magnetic Density: {self.magnetic_density.get()} g/cm³")
        print(f"  Roughness: {self.roughness.get()} Angstrom")
        print(f"  Previous Roughness: {self.prev_roughness.get()} Angstrom")
        print(f"  Linked Previous Roughness: {self.linked_prev_roughness}")
        if self.magnetic:
            print(f"  Magnetic: Yes, Direction: {self.magnetic_direction}")
        else:
            print("  Magnetic: No")

        for detail in self.compound_details:
            print(
                f"  Detail - {detail.index}: {detail.name} - "
                f"{detail.stochiometric_fraction} "
                f"{detail.thickness.get()} Angstrom "
                f"{detail.roughness.get()} Angstrom "
                f"{detail.prev_roughness.get()} Angstrom "
                f"{detail.molar_density.get()} mol/cm³ "
                f"{detail.molar_magnetic_density.get()} mol/cm³"
            )


def get_compound_details(formula: str, atoms: list[Atom]) -> list[CompoundDetails]:
    """Parse formula and create compound details.

    Args:
        formula: Chemical formula to parse
        atoms: Available atom objects

    Returns:
        Details for each element in formula
    """
    details = []
    for idx, part in enumerate(formula.split(",")):
        element, count = part.split(":")
        atom = find_atom(element, atoms)
        if atom is None:
            raise ValueError(f"Unknown atom: {element}")

        detail = CompoundDetails()
        detail.index = idx
        detail.name = element
        detail.stochiometric_fraction = int(count)
        detail.atom = atom
        details.append(detail)

    return details


def create_compound(
    parameters_container: ParametersContainer,
    name: str,
    formula: str,
    thickness: float,
    density: float,
    atoms: list[Atom],
    roughness: float = 0.0,
    prev_roughness: float = 0.0,
    linked_prev_roughness: bool = False,
    magetic_density: float = 0.0,
) -> Compound:
    """Create compound with parameters and atomic details.

    Args:
        parameters_container: Container for parameter management
        name: Compound name
        formula: Chemical formula
        thickness: Layer thickness
        density: Material density
        atoms: Available atom objects
        roughness: Interface roughness
        prev_roughness: Previous interface roughness
        linked_prev_roughness: Link previous roughness if True
        magetic_density: Magnetic density

    Returns:
        Configured compound object
    """
    compound_details = get_compound_details(formula, atoms)

    total_mass = 0.0
    for detail in compound_details:
        total_mass += detail.atom.mass * detail.stochiometric_fraction

    com = Compound()
    com.name = name
    com.formula = formula
    com.thickness = parameters_container.new_parameter(f"{name}-thickness", thickness)
    com.density = parameters_container.new_parameter(f"{name}-density", density)
    com.magnetic_density = parameters_container.new_parameter(f"{name}-magnetic_density", magetic_density)
    com.roughness = parameters_container.new_parameter(f"{name}-roughness", roughness)
    com.prev_roughness = parameters_container.new_parameter(f"{name}-prev_roughness", prev_roughness)
    com.linked_prev_roughness = linked_prev_roughness

    for detail in compound_details:
        frac_i = detail.stochiometric_fraction / total_mass
        base_name = f"{name}-{detail.name}"

        detail.molar_density = parameters_container.register_derived(
            f"{base_name}-molar_density",
            lambda frac=frac_i: com.density.get() * frac,
        )
        detail.molar_magnetic_density = parameters_container.register_derived(
            f"{base_name}-molar_magnetic_density",
            lambda frac=frac_i: com.magnetic_density.get() * frac,
        )
        detail.roughness = com.roughness
        detail.thickness = com.thickness
        detail.prev_roughness = com.prev_roughness

    com.compound_details = compound_details
    return com
