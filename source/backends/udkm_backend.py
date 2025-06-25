import numpy as np
import udkm1Dsim as ud
from source.core.structure import Structure
from source.core.layer      import Layer

u = ud.u                                  # pint registry

def _layer_to_ud(layer: Layer, E_eV: float) -> ud.AmorphousLayer:
    atom = ud.Atom(layer.name)

    print(layer.id, layer.name)

    return ud.AmorphousLayer(
        layer.id,
        layer.name,
        layer.thickness*u.nm,
        layer.density*u.kg/u.m**3,
        atom=atom,
        # opt_ref_index = layer.get_n(E_eV)
    )

def _to_ud_structure(stack: Structure, E_eV: float) -> ud.Structure:
    S = ud.Structure(stack.name)
    print(len(stack.layers))
    for L in stack.layers:
        S.add_sub_structure(_layer_to_ud(L, E_eV), 1)
    return S

def reflectivity(stack: Structure, qz: np.ndarray, E_eV: float) -> tuple[np.ndarray, np.ndarray]:
    S   = _to_ud_structure(stack, E_eV)
    dyn = ud.XrayDyn(S, True)
    dyn.disp_messages = True
    dyn.save_data = False

    dyn.energy = np.r_[E_eV] * u.eV
    dyn.qz     = qz / u.angstrom

    a = dyn.homogeneous_reflectivity()
    # return R_sigma, R_pi
