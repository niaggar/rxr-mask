"""UDKM1Dsim backend for X-ray reflectometry calculations.

Provides integration with the UDKM1Dsim library (currently under development).
"""

import numpy as np
import udkm1Dsim as ud
# from source.core.structure import Structure
# from source.core.compound import Compound
# from source.pint_init import ureg

# from dask.distributed import Client

def test():
    """Test function to verify backend module loading."""
    print("Udkm1Dsim backend module loaded successfully.")

# def _compound_to_ud(compound: Compound) -> list[ud.AmorphousLayer]:
#     ud_layers = []

#     for i in reversed(range(len(compound.layers))):
#         level_layer = compound.layers[i]
#         atom_mixed = ud.AtomMixed(compound.formula)

#         for atomic_layer in level_layer:
#             if atomic_layer.atom.ff.ff_path is None:
#                 raise ValueError(f"Layer {atomic_layer.id} has no form factor valid path to use udkm1Dsim.")

#             atom_symbol = atomic_layer.atom.symbol
#             atom = ud.Atom(atom_symbol, atomic_form_factor_path=atomic_layer.atom.ff.ff_path)
#             atom_mixed.add_atom(atom, atomic_layer.stochiometric_fraction)

#         ud_layer = ud.AmorphousLayer(
#             compound.id,
#             f"{compound.name} layer",
#             compound.get_thickness_layer(i) * ureg.angstrom,
#             compound.base_density * ureg.g / ureg.cm**3,
#             atom=atom_mixed,
#         )

#         ud_layers.append(ud_layer)

#     return ud_layers


# def _to_ud_structure(stack: Structure) -> ud.Structure:
#     ud_stack = ud.Structure(stack.name)
    
#     for comp in reversed(stack.compounds):
#         if not isinstance(comp, Compound):
#             raise TypeError("Expected a Compound instance in the stack.")

#         sub_structure = ud.Structure(comp.name)
#         layers = _compound_to_ud(comp)
        
#         for layer in layers:
#             sub_structure.add_sub_structure(layer)

#         ud_stack.add_sub_structure(sub_structure)

#     return ud_stack


# def reflectivity(stack: Structure, qz: np.ndarray, E_eV: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     ud_structure = _to_ud_structure(stack)
#     # ud_structure.visualize()

#     dyn_mag = ud.XrayDynMag(ud_structure, True, disp_messages=False, save_data=False)
#     dyn_mag.energy = np.r_[E_eV] * ureg.eV
#     dyn_mag.qz = qz / ureg.angstrom

#     dyn_mag.set_polarization(0, 3)
#     R_hom_sigma, _, _, _ = dyn_mag.homogeneous_reflectivity()

#     dyn_mag.set_polarization(0, 4)
#     R_hom_pi, _, _, _ = dyn_mag.homogeneous_reflectivity()

#     return dyn_mag.qz[0, :], R_hom_sigma[0, :], R_hom_pi[0, :]

# # TODO: Think how to implement this function
# # def reflectivity_parallel(stack: Structure, qz: np.ndarray, E_eV: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
# #     ud_structure = _to_ud_structure(stack)
# #     # ud_structure.visualize()

# #     dyn_mag = ud.XrayDynMag(ud_structure, True, disp_messages=False, save_data=False)
# #     dyn_mag.energy = np.r_[E_eV] * ureg.eV
# #     dyn_mag.qz = qz / ureg.angstrom

# #     dclient = Client()
# #     dyn_mag.set_polarization(0, 3)
# #     R_hom_sigma, _, _, _ = dyn_mag.homogeneous_reflectivity(calc_type="parallel", client=dclient)

# def energy_scan(stack: Structure, E_eVs: list[float], theta_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     ud_structure = _to_ud_structure(stack)
#     # ud_structure.visualize()

#     dyn_mag = ud.XrayDynMag(ud_structure, True, disp_messages=False, save_data=False)
#     dyn_mag.energy = np.array(E_eVs) * ureg.eV
#     dyn_mag.qz = np.sin(np.radians(theta_deg)) * (E_eVs[0] * 0.001013546143) / ureg.angstrom

#     dyn_mag.set_polarization(0, 3)
#     R_hom_sigma, _, _, _ = dyn_mag.homogeneous_reflectivity()

#     dyn_mag.set_polarization(0, 4)
#     R_hom_pi, _, _, _ = dyn_mag.homogeneous_reflectivity()

#     return np.array(E_eVs), R_hom_sigma, R_hom_pi
