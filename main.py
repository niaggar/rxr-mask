import numpy as np
from matplotlib import pyplot as plt

from backends import pr_backend, udkm_backend
from core.atom import Atom
from core.layer import Layer
from core.structure import Structure




Cr = Atom(24, "Cr", 52.00)
Si = Atom(14, "Si", 28.09)

lay_Cr = Layer("Id-Cr", "Cr", 1, 7140, atom=Cr)
lay_Si = Layer("Id-Si", "Si", 1, 2336, atom=Si)

sample = Structure("Cr/Si stack")
sample.add(lay_Cr, repeat=10)
sample.add(lay_Si, repeat=200)


E_eV = 708.0                                   # energía de prueba
qz    = np.linspace(0.01, 5.0, 1500)           # Å⁻¹

# R_phi, R_pi = pr_backend.reflectivity(sample, qz, E_eV)
#
#
# plt.semilogy(qz, R_phi, label="σ-pol")
# plt.semilogy(qz, R_pi, "--", label="π-pol")
# plt.xlabel(r"$q_z$ (Å$^{-1}$)")
# plt.ylabel("Reflectividad")
# plt.title("PythonReflectivity – Cr/Si")
# plt.legend(); plt.tight_layout(); plt.show()


R_phi, R_pi = udkm_backend.reflectivity(sample, qz, E_eV)


plt.semilogy(qz, R_phi, label="σ-pol")
plt.semilogy(qz, R_pi, "--", label="π-pol")
plt.xlabel(r"$q_z$ (Å$^{-1}$)")
plt.ylabel("Reflectividad")
plt.title("PythonReflectivity – Cr/Si")
plt.legend(); plt.tight_layout(); plt.show()


