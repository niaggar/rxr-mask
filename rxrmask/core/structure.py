from rxrmask.core import Compound, AtomLayer, Layer
from rxrmask.core.parameter import Parameter, ParametersContainer
from rxrmask.utils import get_density_profile_from_element_data

from dataclasses import dataclass, field


@dataclass
class Structure:
    """Complete multilayer structure for X-ray reflectometry.

    Attributes:
        name: Structure name
        params_container: Container for parameters
        n_compounds: Number of compounds
        n_layers: Number of discretized layers
        layers: Discretized layer objects
        compounds: Compound objects
        element_data: Cached element data for optimization
        atoms: Atom objects cache
        step: Discretization step size in Angstroms
    """

    name: str
    params_container: ParametersContainer
    n_compounds: int = 0
    n_layers: int = 0
    layers: list[Layer] = field(default_factory=list)
    compounds: list[Compound] = field(default_factory=list)

    element_data: dict | None = None
    atoms: dict | None = None
    step: float = 0.1

    def __init__(self, name: str, n_compounds: int, params_container: ParametersContainer):
        self.name = name
        self.n_compounds = n_compounds
        self.compounds = [None] * n_compounds  # type: ignore
        self.params_container = params_container

    def add_compound(self, index: int, compound: Compound) -> None:
        """Add compound at specified index."""
        if index < 0 or index >= self.n_compounds:
            raise IndexError("Index out of range for compounds.")
        if not isinstance(compound, Compound):
            raise TypeError("Expected a Compound instance.")

        self.compounds[index] = compound



    def validate_compounds(self):
        """Validate that all compounds are defined and link roughness if needed."""
        for compound in self.compounds:
            if compound is None:
                raise ValueError("All compounds must be defined.")

        for i in range(1, len(self.compounds)):
            current_compound = self.compounds[i]
            if current_compound.linked_prev_roughness:
                prev_compound = self.compounds[i - 1]
                current_compound.prev_roughness.fit = False
                current_compound.prev_roughness.independent = False
                current_compound.prev_roughness.depends_on = prev_compound.roughness

                if len(current_compound.compound_details) != len(prev_compound.compound_details):
                    raise ValueError(f"Compound {i} has different number of elements than compound {i-1}.")

                for j, detail in enumerate(current_compound.compound_details):
                    detail.prev_roughness.fit = False
                    detail.prev_roughness.independent = False
                    detail.prev_roughness.depends_on = prev_compound.compound_details[j].roughness



    def create_layers(self, step: float = 0.1, track_layers_params=False) -> None:
        """Create discretized layers from compounds with specified step size."""
        self.step = step
        self.element_data, self.atoms = self._create_element_data()

        z, dens, m_dens = get_density_profile_from_element_data(self.element_data, self.step)

        layers = []
        for i in range(1, len(z)):
            thickness = z[i] - z[i - 1]

            elements = []
            for element_name, atom in self.atoms.items():
                molar_density = dens[element_name][i] if element_name in dens else 0.0
                molar_magnetic_density = m_dens[element_name][i] if element_name in m_dens else 0.0

                temp_name = f"layer_{i}_{element_name}"
                m_dens_param = None
                m_magnetic_density_param = None
                if track_layers_params:
                    m_dens_param = self.params_container.new_parameter(f"{temp_name}_density", molar_density, fit=True)
                    m_magnetic_density_param = self.params_container.new_parameter(f"{temp_name}_mag_density", molar_magnetic_density, fit=True)
                else:
                    m_dens_param = Parameter(
                        id=0,
                        name=f"{temp_name}_density",
                        value=molar_density,
                        fit=False,
                        lower=None,
                        upper=None,
                    )
                    m_magnetic_density_param = Parameter(
                        id=0,
                        name=f"{temp_name}_mag_density",
                        value=molar_magnetic_density,
                        fit=False,
                        lower=None,
                        upper=None,
                    )

                element_layer = AtomLayer(
                    atom=atom,
                    molar_density=m_dens_param,
                    molar_magnetic_density=m_magnetic_density_param,
                )
                elements.append(element_layer)

            layer = Layer(id=f"Layer_{i}", thickness=thickness, elements=elements)
            layers.append(layer)

        self.layers = layers
        self.n_layers = len(layers)

    def update_layers(self) -> None:
        """Update layer parameters based on current element data and density profile."""
        z, dens, m_dens = get_density_profile_from_element_data(self.element_data, self.step)

        if len(z) != len(self.layers) + 1:
            raise ValueError("Inconsistent current layer count with density profile.")

        for i in range(1, len(z)):
            layer_idx = i - 1
            if layer_idx >= len(self.layers):
                break

            layer = self.layers[layer_idx]
            for element_layer in layer.elements:
                atom_name = element_layer.atom.name

                if atom_name in dens:
                    new_density = dens[atom_name][i]
                    element_layer.molar_density.set(new_density)

                if atom_name in m_dens:
                    new_mag_density = m_dens[atom_name][i]
                    if element_layer.molar_magnetic_density is not None:
                        element_layer.molar_magnetic_density.set(new_mag_density)

    def _create_element_data(self):
        """Create element data structure from compounds for density calculations."""
        for compound in self.compounds:
            if compound is None:
                raise ValueError("All compounds must be defined to get element data.")

        atoms = {}
        data = {}

        for i, compound in enumerate(self.compounds):
            if compound is None:
                raise ValueError("All compounds must be defined to get element data.")

            for element in compound.compound_details:
                name = element.name

                if name not in atoms:
                    atoms[name] = element.atom
                    data[name] = {
                        "density_params": [None] * (len(self.compounds) + 1),
                        "magnetic_density_params": [None] * (len(self.compounds) + 1),
                        "roughness_params": [None] * len(self.compounds),
                        "thickness_params": [x.compound_details[element.index].thickness for x in self.compounds],
                    }

                data[name]["density_params"][i] = element.molar_density
                data[name]["magnetic_density_params"][i] = element.molar_magnetic_density

                if i > 0:
                    if data[name]["roughness_params"][i - 1] is None:
                        data[name]["roughness_params"][i - 1] = element.prev_roughness

                data[name]["roughness_params"][i] = element.roughness

        return data, atoms

    def print_details(self):
        print(f"Structure: {self.name}")
        print(f"Number of compounds: {self.n_compounds}")
        print(f"Number of layers: {self.n_layers}")

        for i, compound in enumerate(self.compounds):
            if compound is None:
                continue
            print(f"Compound {i}: {compound.name}")
            print(f"  Thickness: {compound.thickness} Angstrom")
            print(f"  Density: {compound.density} g/cm³")
            if compound.magnetic_density:
                print(f"  Magnetic Density: {compound.magnetic_density} mol/cm³")
            print(f"  Roughness: {compound.roughness} Angstrom")
            print(f"  Previous Roughness: {compound.prev_roughness} Angstrom")
