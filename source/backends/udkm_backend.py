import numpy as np
import udkm1Dsim as ud
from source.core.structure import Structure
from source.core.compound import Compound
from source.core.layer import Layer

# Pint registry from udkm1Dsim
u = ud.u

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _layer_to_ud(layer: Layer, E_eV: float) -> ud.AmorphousLayer:
    # Build the required `udkm1Dsim.Atom` instance from the atomic symbol
    atom_symbol = layer.atom.symbol
    atom = ud.Atom(atom_symbol, atomic_form_factor_path=None)

    # Thickness: Å → nm for pint compatibility
    thickness = (layer.thickness * 0.1) * u.nm

    # Density already in kg m⁻³ → attach pint unit
    density = layer.density * u.kg / u.m**3

    # Complex refractive index (optional but speeds calculations)
    n_complex = layer.get_n(E_eV)

    return ud.AmorphousLayer(
        layer.id,
        f"{atom_symbol} layer",
        thickness,
        density,
        atom=atom,
        opt_ref_index=n_complex,
    )


def _compound_to_ud(compound: Compound, E_eV: float) -> list[ud.AmorphousLayer]:
    """Flatten a `Compound` into a list of udkm1Dsim layers respecting `n_layers`."""
    ud_layers: list[ud.AmorphousLayer] = []
    for i in range(compound.n_layers):
        for sublayer in compound.layers[i]:
            ud_layers.append(_layer_to_ud(sublayer, E_eV))
    return ud_layers


def _to_ud_structure(stack: Structure, E_eV: float) -> ud.Structure:
    """Transform the whole generic `Structure` into an udkm1Dsim `Structure`."""
    ud_stack = ud.Structure(stack.name)
    for comp in stack.compounds:
        for ud_layer in _compound_to_ud(comp, E_eV):
            # Add each physical layer once; for repeated super‑lattices simply call
            # `Structure.add` with repeat >1 upstream.
            ud_stack.add_sub_structure(ud_layer, 1)
    return ud_stack

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def reflectivity(stack: Structure, qz: np.ndarray, E_eV: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_sigma, R_pi) for the given *stack* at momentum *qz* (Å⁻¹).

    At the moment `udkm1Dsim` does not expose σ/π separation in the homogeneous
    solver – we therefore return the total reflectivity for both channels so the
    calling code keeps a consistent interface with `pr_backend.reflectivity`.
    """
    ud_structure = _to_ud_structure(stack, E_eV)

    dyn = ud.XrayDyn(ud_structure, relativistic=False)  # non‑magnetic case
    dyn.disp_messages = False
    dyn.save_data = False

    dyn.energy = np.r_[E_eV] * u.eV
    dyn.qz = qz / u.angstrom  # input qz comes in Å⁻¹

    R, _, _, _ = dyn.homogeneous_reflectivity()
    R_total = R[0, :]
    return R_total, R_total  # placeholder until polarisation split is available
