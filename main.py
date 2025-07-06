from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from source.core.atom import Atom
from source.core.formfactor import FormFactorModel
from source.core.structure import Structure
from source.core.compound import create_compound




@dataclass
class FormFactorData(FormFactorModel):
    ff_data: pd.DataFrame | None = field(default=None)

    def __init__(self, ff_path: str):
        super().__init__(ff_path)
        self.read_data()

    def read_data(self):
        if self.ff_path is None:
            raise ValueError("Form factor path is not set.")
        
        file = Path(self.ff_path)
        if not file.exists():
            raise FileNotFoundError(f"Form factor data file '{self.ff_path}' does not exist.")
        
        self.ff_data = pd.read_csv(
            file,
            delimiter="\t",
            header=None,
            index_col=False,
            names=["E", "f1", "f2"],
            dtype=float,
            comment="#",
        )

    def get_f1f2(self, energy_eV: float, *args) -> tuple[float, float]:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        f1 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f1))
        f2 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f2))
        
        return f1, f2
    
    def get_all_f1f2(self, *args) -> pd.DataFrame:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        return self.ff_data




cr_ff = FormFactorData(ff_path="./source/materials/form_factor/Cr.txt")
si_ff = FormFactorData(ff_path="./source/materials/form_factor/Si.txt")
cr_atom = Atom(
    Z=24,
    name="Cr",
    symbol="Cr",
    mass=51.9961,
    ff=cr_ff,
)
si_atom = Atom(
    Z=14,
    name="Si",
    symbol="Si",
    mass=28.0855,
    ff=si_ff,
)


comp_cr = create_compound(
    id="Cr",
    name="Cr",
    thickness=10.0,
    density=7140,
    formula="Cr:1",
    n_layers=1,
    atoms_prov=[cr_atom],
)
comp_si = create_compound(
    id="Si",
    name="Si",
    thickness=200,
    density=2336,
    formula="Si:1",
    n_layers=1,
    atoms_prov=[si_atom],
)


struc = Structure(name="Cr/Si")
struc.add(comp_cr, repeat=1)
struc.add(comp_si, repeat=1)


import source.backends.pr_backend as pr_backend


E_eV = 708.0
qz = np.linspace(0.01, 5.0, 1500)
R_phi, R_pi = pr_backend.reflectivity(struc, qz, E_eV)


plt.semilogy(qz, R_phi, label="σ-pol")
plt.semilogy(qz, R_pi, "--", label="π-pol")
plt.xlabel(r"$q_z$ (Å$^{-1}$)")
plt.ylabel("Reflectividad")
plt.title("PythonReflectivity – Cr/Si")
plt.legend()
plt.tight_layout()
plt.show()

