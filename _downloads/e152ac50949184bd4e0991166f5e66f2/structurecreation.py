"""
Structure Creation Example
===========================

This example demonstrates how to create a structure with a single compound and a single atom.
"""
from rxrmask.core import Atom, Structure, FormFactorLocalDB, ParametersContainer
from rxrmask.core import create_compound


mn_ff = FormFactorLocalDB(element="Mn", is_magnetic=False)
mn_atom = Atom(
    Z=25,
    name="Mn",
    ff=mn_ff,
)

parameters_container = ParametersContainer()
comp = create_compound(
    parameters_container=parameters_container,
    name="Manganese",
    formula="Mn:1",
    thickness=50.0,
    density=5.12,
    atoms=[mn_atom],
    roughness=0.0,
)

struc = Structure(name=f"Test Structure", n_compounds=1, params_container=parameters_container)
struc.add_compound(0, comp)
struc.validate_compounds()
struc.create_layers(step=0.1)
