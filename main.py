# from ctypes import Structure
# import numpy as np
# from matplotlib import pyplot as plt

# from source.core.atom import Atom
# from source.core.layer import Layer



# Cr = Atom(24, "Cr", 52.00)
# Si = Atom(14, "Si", 28.09)

# lay_Cr = Layer("Id-Cr", "Cr", 1, 7140, atom=Cr)
# lay_Si = Layer("Id-Si", "Si", 1, 2336, atom=Si)

# sample = Structure("Cr/Si stack")
# sample.add(lay_Cr, repeat=10)
# sample.add(lay_Si, repeat=200)






import numpy as np
from matplotlib import pyplot as plt

from source.core.structure import Structure
from source.core.compound import create_compound


comp_cr = create_compound(
    id="Cr",
    name="Cr",
    thickness=10.0,  # en u.nm
    density=7140,      # en u.kg/u.m**3
    formula="Cr:1",  # Cr y Si en la misma capa
    n_layers=10,       # número de capas
)
comp_si = create_compound(
    id="Si",
    name="Si",
    thickness=200,  # en u.nm
    density=2336,      # en u.kg/u.m**3
    formula="Si:1",  # Cr y Si en la misma capa
    n_layers=200,       # número de capas
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

