# %%
import numpy as np
import time

from rxrmask.core import (
    Atom,
    Structure,
    FormFactorLocalDB,
    ParametersContainer,
    SimReflectivityData,
    ReflectivityScan,
    create_compound,
)
from rxrmask.utils import (
    plot_reflectivity,
)
from rxrmask.backends import (
    PRReflectivityBackend,
)
from rxrmask.optimization import (
    FitContext
)

# %%
o_ff = FormFactorLocalDB(element="O", is_magnetic=False)
sr_ff = FormFactorLocalDB(element="Sr", is_magnetic=False)
ti_ff = FormFactorLocalDB(element="Ti", is_magnetic=False)
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

parameters_container = ParametersContainer()
crystal = create_compound(
    parameters_container=parameters_container,
    name="SrTiO3-crystal",
    formula="Sr:1,Ti:1,O:3",
    thickness=50.0,
    density=5.12,
    atoms=[sr_atom, ti_atom, o_atom],
    roughness=0.0,
    prev_roughness=0.0,
    linked_prev_roughness=False,
)
film = create_compound(
    parameters_container=parameters_container,
    name="SrTiO3-film",
    formula="Sr:1,Ti:1,O:3",
    thickness=23.0,
    density=5.52,
    atoms=[sr_atom, ti_atom, o_atom],
    roughness=4.0,
    prev_roughness=0.0,
    linked_prev_roughness=False,
)

struc = Structure(name="SrTiO3-struc", n_compounds=2, params_container=parameters_container)
struc.add_compound(0, crystal)
struc.add_compound(1, film)
struc.validate_compounds()
struc.create_layers(step=0.1)

crystal.roughness.fit = True
film.roughness.fit = True
film.prev_roughness.fit = True
film.thickness.fit = True
film.density.fit = True

# %%
init_params = parameters_container.get_fit_vector()
print(f"Initial parameters: {init_params}")
for param in parameters_container.parameters:
    if param.fit:
        print(f"{param.name}: {param.value}")

# %%
backend = PRReflectivityBackend()

E_eV = 600
Theta = np.linspace(0.1, 89.1, num=1001)
qz = np.sin(Theta * np.pi / 180) * (E_eV * 0.001013546143)
initial_ref = backend.compute_reflectivity(struc, qz, E_eV)
plot_reflectivity(initial_ref.qz, initial_ref.R_s, initial_ref.R_p, initial_ref.energy, "")


# %%


optimizer = LeastSquaresOptimizer()
fitter = RXRFitter(model, experimental_data, optimizer)


init_params = model.parameters_container.get_fit_vector()
print(f"Initial parameters: {init_params}")
for param in model.parameters_container.parameters:
    if param.fit:
        print(f"{param.name}: {param.value}")

start_time = time.time()
result = fitter.fit(init_params, [])
end_time = time.time()

print(f"Fitting time: {end_time - start_time:.2f} seconds")
print(f"Fitting result: {result}")
print(f"Fitted parameters: {model.parameters_container.get_fit_vector()}")


final_ref = model.compute_reflectivity(qz, E_eV)
plot_reflectivity(final_ref.qz, final_ref.R_s, final_ref.R_p, final_ref.energy, "")


# In[13]:


plot_reflectivity(experimental_data.qz, experimental_data.R_s, final_ref.R_s, final_ref.energy, "")
plot_reflectivity(experimental_data.qz, experimental_data.R_p, final_ref.R_p, final_ref.energy, "")
