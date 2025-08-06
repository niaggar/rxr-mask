"""Multilayer structure representation for X-ray reflectometry.

Provides Structure class for managing compounds and layer discretization.
"""

from rxrmask.core import Compound, AtomLayer, Layer
from rxrmask.core.parameter import ParametersContainer
from rxrmask.utils import get_density_profile_from_element_data

from dataclasses import dataclass, field


@dataclass
class Structure:
    """Complete multilayer structure for X-ray reflectometry.

    Manages compounds organization, layer discretization, and density profiles.

    Attributes:
        name (str): Structure name.
        n_compounds (int): Number of compounds.
        n_layers (int): Number of discretized layers.
        layers (list[Layer]): Discretized layer objects.
        compounds (list[Compound | None]): Compound objects.
        element_data (dict | None): Cached element data for optimization.
        layer_thickness_params (list | None): Layer thickness parameters.
        atoms (dict | None): Atom objects cache.
    """

    name: str
    params_container: ParametersContainer
    n_compounds: int = 0
    n_layers: int = 0
    layers: list[Layer] = field(default_factory=list)
    compounds: list[Compound] = field(default_factory=list)

    element_data: dict | None = None
    layer_thickness_params: list | None = None
    atoms: dict | None = None
    step: float = 0.1

    def __init__(
        self, name: str, n_compounds: int, params_container: ParametersContainer
    ):
        """Initialize structure with name and number of compounds."""
        self.name = name
        self.n_compounds = n_compounds
        self.compounds = [None] * n_compounds # type: ignore
        self.params_container = params_container

    def add_compound(self, index: int, compound: Compound) -> None:
        """Add compound at specified index."""
        if index < 0 or index >= self.n_compounds:
            raise IndexError("Index out of range for compounds.")
        if not isinstance(compound, Compound):
            raise TypeError("Expected a Compound instance.")

        self.compounds[index] = compound

    def validate_compounds(self):
        """Validate that all compounds are defined."""
        for compound in self.compounds:
            if compound is None:
                raise ValueError("All compounds must be defined.")

        for i in range(1, len(self.compounds)):
            current_compound = self.compounds[i]  # type: ignore
            if current_compound.linked_prev_roughness:
                prev_compound = self.compounds[i - 1]  # type: ignore
                current_compound.prev_roughness = prev_compound.roughness

                if len(current_compound.compound_details) != len(
                    prev_compound.compound_details
                ):
                    raise ValueError(
                        f"Compound {i} has different number of elements than compound {i-1}."
                    )
                
                for j, detail in enumerate(current_compound.compound_details):
                    detail.prev_roughness = prev_compound.compound_details[j].roughness

    def create_layers(self, step: float = 0.1) -> None:
        """Create discretized layers from compounds with specified step size."""
        self.step = step
        self.element_data, self.layer_thickness_params, self.atoms = (
            self._create_element_data()
        )

        z, dens, m_dens, _ = get_density_profile_from_element_data(
            self.element_data, self.layer_thickness_params, self.atoms, self.step
        )

        layers = []
        for i in range(1, len(z)):
            thickness = z[i] - z[i - 1]

            elements = []
            for element_name, atom in self.atoms.items():
                molar_density = dens[element_name][i] if element_name in dens else 0.0
                molar_magnetic_density = (
                    m_dens[element_name][i] if element_name in m_dens else 0.0
                )

                molar_density_param = self.params_container.new_parameter(
                    f"layer_{i}_{element_name}_density", molar_density
                )
                molar_magnetic_density_param = self.params_container.new_parameter(
                    f"layer_{i}_{element_name}_mag_density", molar_magnetic_density
                )

                element_layer = AtomLayer(
                    atom=atom,
                    molar_density=molar_density_param,
                    molar_magnetic_density=molar_magnetic_density_param,
                )
                elements.append(element_layer)

            layer = Layer(id=f"Layer_{i}", thickness=thickness, elements=elements)
            layers.append(layer)

        self.layers = layers
        self.n_layers = len(layers)

    def update_layers(self) -> None:
        """Update existing layers with current parameter values."""
        if not self.layers:
            raise ValueError(
                "No layers exist. Call create_layers or create_layers_optimized first."
            )

        z, dens, m_dens, _ = get_density_profile_from_element_data(
            self.element_data, self.layer_thickness_params, self.atoms, self.step
        )

        if len(z) != len(self.layers) + 1:
            raise ValueError(
                f"Length of z ({len(z)}) does not match number of layers ({len(self.layers)})."
            )

        for i, layer in enumerate(self.layers):
            layer_index = i + 1

            if layer_index < len(z):
                new_thickness = (
                    z[layer_index] - z[layer_index - 1]
                    if layer_index > 0
                    else self.step
                )
                layer.thickness = new_thickness

                # Update density values for each element in the layer
                for element_layer in layer.elements:
                    element_name = element_layer.atom.name

                    if element_name in dens and layer_index < len(dens[element_name]):
                        # Update regular density
                        new_density = dens[element_name][layer_index]
                        element_layer.molar_density.set(new_density)

                        # Update magnetic density
                        if element_name in m_dens and layer_index < len(
                            m_dens[element_name]
                        ):
                            new_mag_density = m_dens[element_name][layer_index]
                            if element_layer.molar_magnetic_density is not None:
                                element_layer.molar_magnetic_density.set(
                                    new_mag_density
                                )

    def _create_element_data(self):
        """Create element data structure from compounds for density calculations."""
        for compound in self.compounds:
            if compound is None:
                raise ValueError("All compounds must be defined to get element data.")

        atoms = {}
        data = {}

        for i, compound in enumerate(self.compounds):
            if compound is None:
                continue

            for j, element in enumerate(compound.compound_details):
                name = element.name

                if name not in atoms:
                    atoms[name] = element.atom
                    data[name] = {
                        "density_params": [None] * (len(self.compounds) + 1),
                        "magnetic_density_params": [None] * (len(self.compounds) + 1),
                        "roughness_params": [None] * len(self.compounds),
                        "thickness_params": [None] * len(self.compounds),
                    }

                data[name]["density_params"][i] = element.molar_density
                data[name]["magnetic_density_params"][i] = element.molar_magnetic_density
                
                data[name]["thickness_params"][i] = element.thickness
                
                if i > 0:
                    if data[name]["roughness_params"][i - 1] is None:
                        data[name]["roughness_params"][i - 1] = element.prev_roughness

                data[name]["roughness_params"][i] = element.roughness
                
                
        for element_name, element_info in data.items():
            print(f"  {element_name}:")
            for param_name, param_value in element_info.items():
                print(f"    {param_name}: {[p.get() if p else None for p in param_value]}")

        layer_thickness_params = [
            compound.thickness for compound in self.compounds if compound is not None
        ]
        return data, layer_thickness_params, atoms

    def print_details(self):
        """Print details of the structure."""
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