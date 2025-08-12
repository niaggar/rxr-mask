# %%
import numpy as np
import time
from rxrmask.core import (
    Atom,
    Structure,
    FormFactorLocalDB,
    ParametersContainer,
    FormFactorModel,
    FormFactorVacancy,
    SimReflectivityData,
    create_compound,
    load_energy_scans,
    load_reflectivity_scans,
)
from rxrmask.utils import (
    plot_reflectivity,
    plot_energy_scan,
    plot_density_profile,
    get_density_profile_from_element_data,
)
from rxrmask.backends import (
    PRReflectivityBackend,
    PRParallelReflectivityBackend,
)

def set_compound_params(structure: Structure, fit: bool):
    for compound in structure.compounds:
        compound.thickness.fit = fit
        compound.roughness.fit = fit
        compound.prev_roughness.fit = fit
        compound.density.fit = fit
        compound.magnetic_density.fit = fit
# %%
mn_ff = FormFactorLocalDB(element="Mn", is_magnetic=False)
o_ff = FormFactorLocalDB(element="O", is_magnetic=False)
sr_ff = FormFactorLocalDB(element="Sr", is_magnetic=False)
ti_ff = FormFactorLocalDB(element="Ti", is_magnetic=False)
c_ff = FormFactorLocalDB(element="C", is_magnetic=False)
la_ff = FormFactorLocalDB(element="La", is_magnetic=False)

la_atom = Atom(
    Z=57,
    name="La",
    ff=la_ff,
)
mn_atom = Atom(
    Z=25,
    name="Mn",
    ff=mn_ff,
)
o_atom = Atom(
    Z=8,
    name="O",
    ff=o_ff,
)
sr_atom = Atom(
    Z=38,
    name="Sr",
    ff=sr_ff,
)
ti_atom = Atom(
    Z=22,
    name="Ti",
    ff=ti_ff,
)
c_atom = Atom(
    Z=6,
    name="C",
    ff=c_ff,
)
x1_atom = Atom(
    Z=0,
    name="X1",
    ff=c_ff,
)
x2_atom = Atom(
    Z=0,
    name="X2",
    ff=c_ff,
)
# %%
parameters_container = ParametersContainer()
comp_SrTiO3 = create_compound(
    parameters_container=parameters_container,
    name="SrTiO3",
    formula="Sr:1,Ti:1,O:3",
    thickness=50.0,
    density=5.12,
    atoms=[sr_atom, ti_atom, o_atom],
    roughness=0.0,
    prev_roughness=0.0,
    linked_prev_roughness=False,
)
comp_LaMnO3 = create_compound(
    parameters_container=parameters_container,
    name="LaMnO3",
    formula="La:1,Mn:1,O:3",
    thickness=8.0,
    density=5.52,
    atoms=[la_atom, mn_atom, o_atom],
    roughness=4.0,
    prev_roughness=0.0,
    linked_prev_roughness=False,
)
comp_CCO = create_compound(
    parameters_container=parameters_container,
    name="CO2",
    formula="X1:1,X2:1,O:2",
    thickness=15.0,
    density=2.0,
    atoms=[x1_atom, x2_atom, o_atom],
    roughness=1.0,
    prev_roughness=0.0,
    linked_prev_roughness=True,
)

struc = Structure(name=f"Test Structure", n_compounds=3, params_container=parameters_container)
struc.add_compound(0, comp_SrTiO3)
struc.add_compound(1, comp_LaMnO3)
struc.add_compound(2, comp_CCO)
struc.validate_compounds()
struc.create_layers(step=0.1)

model = RXRModel(structure=struc, parameters_container=parameters_container)
set_compound_params(model.structure, fit=True)

# %%

reflectivty_data = load_reflectivity_scans(
    path="/Users/niaggar25/Downloads/Experimental RXR data/B076_AScans.dat",
    initial_name="test_data",
)
energy_data = load_energy_scans(
    path="/Users/niaggar25/Downloads/Experimental RXR data/B074_EScans.dat",
    initial_name="test_data",
)

i = 10
plot_reflectivity(
    reflectivty_data[i].qz,
    np.log(reflectivty_data[i].R),
    np.log(reflectivty_data[i].R),
    reflectivty_data[i].energy_eV,
    reflectivty_data[i].pol,
)
