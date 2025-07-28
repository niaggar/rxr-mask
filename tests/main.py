print("This is a test file for the rxr_mask package.")



from dataclasses import dataclass, field
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from rxrmask.backends import pr_backend
from rxrmask.core.atom import Atom
from rxrmask.core.compound import create_compound
from rxrmask.core.structure import Structure
from rxrmask.core.formfactor import FormFactorModel
from rxrmask.utils.plot import plot_reflectivity, plot_energy_scan




class FormFactorFile(FormFactorModel):
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
            sep="\s+",
            header=None,
            index_col=False,
            names=["E", "f1", "f2"],
            dtype=float,
            comment="#",
        )

    def get_formfactors(self, energy_eV: float, *args) -> tuple[float, float]:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        f1 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f1))
        f2 = float(np.interp(energy_eV, self.ff_data.E, self.ff_data.f2))
        
        return f1, f2
    
    def get_all_formfactors(self, *args) -> pd.DataFrame:
        if self.ff_data is None:
            raise ValueError("Form factor data has not been loaded.")
        
        return self.ff_data
    







mn_ff = FormFactorFile(ff_path="./rxrmask/materials/form_factor/Mn.txt")
o_ff = FormFactorFile(ff_path="./rxrmask/materials/form_factor/O.txt")
sr_ff = FormFactorFile(ff_path="./rxrmask/materials/form_factor/Sr.txt")
ti_ff = FormFactorFile(ff_path="./rxrmask/materials/form_factor/Ti.txt")
c_ff = FormFactorFile(ff_path="./rxrmask/materials/form_factor/C.txt")


la_ff = FormFactorFile(ff_path="./rxrmask/materials/form_factor/La.txt")
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





comp_SrTiO3 = create_compound(
    id="SrTiO3",
    name="SrTiO3",
    thickness=50.0,
    density=5.12,
    roughness=0.0,
    formula="Sr:1,Ti:1,O:3",
    atoms_prov=[sr_atom, ti_atom, o_atom],
)


comp_LaMnO3 = create_compound(
    id="LaMnO3",
    name="LaMnO3",
    thickness=50.0,
    density=6.52,
    formula="La:1,Mn:1,O:3",
    atoms_prov=[la_atom, mn_atom, o_atom],
    roughness=2.0,
)
comp_CCO = create_compound(
    id="CCO",
    name="CCO",
    thickness=10.0,
    density=5,
    formula="C:2,O:1",
    atoms_prov=[c_atom, o_atom],
    roughness=2.0,
)

struc = Structure(name=f"Test Structure", n_compounds=3)
struc.add_compound(0, comp_SrTiO3)
struc.add_compound(1, comp_LaMnO3)
struc.add_compound(2, comp_CCO)
struc.create_layers(step=1)


z, dens, m_dens, atoms = struc.get_density_profile(step=0.1)
plt.figure(figsize=(8, 4))
for name, profile in dens.items():
    plt.plot(z, profile, "-", label=name)
plt.xlabel('Profundidad $z$')
plt.ylabel('Densidad $\\rho(z)$')
plt.ylim(bottom=0)  # Fija el mínimo en y a 0
plt.legend()
plt.tight_layout()
plt.show()




E_eV = 650
Theta = np.linspace(0.1, 89.1, num=1000)
qz = np.sin(Theta * np.pi / 180) * (E_eV * 0.001013546143)





qz_pr, R_phi_pr, R_pi_pr  = pr_backend.reflectivity(struc, qz, E_eV)
# qz_pr_para, R_phi_pr_para, R_pi_pr_para  = pr_backend.reflectivity_parallel(struc, qz, E_eV)
# qz_ud, R_phi_ud, R_pi_ud,  = udkm_backend.reflectivity(struc, qz, E_eV)

plot_reflectivity(qz_pr, R_phi_pr, R_pi_pr, E_eV, "Cr/Si (PR)")
# plot_reflectivity(qz_pr_para, R_phi_pr_para, R_pi_pr_para, E_eV, "Cr/Si (PR)")
# plot_reflectivity(qz_ud, R_phi_ud, R_pi_ud, E_eV, "Cr/Si (UDKM)")



# e_evs = np.linspace(630, 670, num=500).tolist()
# theta_deg = 15

# e_pr, R_phi_pr, R_pi_pr  = pr_backend.energy_scan(struc, e_evs, theta_deg)
# # e_pr_pa, R_phi_pr_pa, R_pi_pr_pa  = pr_backend.energy_scan_parallel(struc, e_evs, theta_deg, n_jobs=-1, use_threads=True, verbose=0)

# plot_energy_scan(e_pr, R_phi_pr, R_pi_pr, theta_deg, "Cr/Si (PR)")
# # plot_energy_scan(e_pr_pa, R_phi_pr_pa, R_pi_pr_pa, theta_deg, "Cr/Si (PR Parallel)")