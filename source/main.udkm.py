# main_udkm.py  –  minimal reflectivity check for udkm1Dsim ≥ 2.0
import udkm1Dsim as ud
u = ud.u

# pint unit registry
import numpy as np
import matplotlib.pyplot as plt
u.setup_matplotlib()

Cr = ud.Atom('Cr')
Si = ud.Atom('Si')

# print("Atoms")
# print(Cr.atomic_form_factor_coeff)
#
# test = Cr.get_atomic_form_factor()

density_Cr = 7140*u.kg/u.m**3
prop_Cr = {}
layer_Cr = ud.AmorphousLayer('Cr', "Cr amorphous", 1*u.nm, density_Cr, atom=Cr, **prop_Cr)


density_Si = 2336*u.kg/u.m**3
prop_Si = {}
layer_Si = ud.AmorphousLayer('Si', "Si amorphous", 1*u.nm, density_Si, atom=Si, **prop_Si)


S = ud.Structure('Fe/Cr AFM Super Lattice')

# create a sub-structure
S.add_sub_structure(layer_Cr, 10)
S.add_sub_structure(layer_Si, 200)

# S.visualize()


dyn_mag = ud.XrayDynMag(S, True)
dyn_mag.disp_messages = True
dyn_mag.save_data = False

dyn_mag.energy = np.r_[600, 708]*u.eV  # set two photon energies
dyn_mag.qz = np.r_[0.01:5:0.01]/u.nm  # qz range
# this is the actual calculation
R_hom, R_hom_phi, _, _ = dyn_mag.homogeneous_reflectivity()
plt.figure()
plt.semilogy(dyn_mag.qz[0, :], R_hom[0, :], label='{}'.format(dyn_mag.energy[0]))
plt.semilogy(dyn_mag.qz[1, :], R_hom[1, :], label='{}'.format(dyn_mag.energy[1]))
plt.ylabel('Reflectivity')
plt.xlabel(r'$q_z$ [1/nm]')
plt.legend()
plt.show()