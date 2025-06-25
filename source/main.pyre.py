import numpy as np
import matplotlib.pyplot as plt
import Pythonreflectivity as pr

E = 833.0                       # energía del rayo-X (eV) –– región Mn-L3
h = 4.135667696e-15             # eV·s
c = 2.99792458e8                # m/s
wavelength = h*c / (E*1e-10)    # λ en Å  (misma conversión que usa el código GO-RXR)

# ------------------------------------------------------------------
# 2.  Construye una estructura de 2 capas (substrato + película fina)
# ------------------------------------------------------------------
A = pr.Generate_structure(2)    # objeto contenedor de Pythonreflectivity:contentReference[oaicite:5]{index=5}

# Capa 0  –– substrato semi-infinito
n_sub = 1 - 1.0e-4 + 1j*5.0e-6  # índice:  n = 1-δ+iβ
A[0].seteps(n_sub**2)           # versión actual todavía acepta ε en vez de χ

# Capa 1  –– película de 40 Å
n_film = 1 - 2.0e-4 + 1j*8.0e-6
A[1].seteps(n_film**2)
A[1].setd(40.0)                 # grosor (Å)

# ------------------------------------------------------------------
# 3.  Calcula reflectividad σ/π vs qz
# ------------------------------------------------------------------
qz = np.linspace(0.01, 0.5, 250)                    # Å⁻¹
Theta = np.arcsin(qz / E / 0.001013546247)*180/np.pi  # mismo factor que usa GO-RXR:contentReference[oaicite:6]{index=6}
Rσ, Rπ = pr.Reflectivity(A, Theta, wavelength, MagneticCutoff=1e-10)  # :contentReference[oaicite:7]{index=7}

print(f"σ(0.02 Å⁻¹) = {Rσ[1]:.3e}")
print(f"π(0.02 Å⁻¹) = {Rπ[1]:.3e}")

# ------------------------------------------------------------------
# 4.  Traza la curva (opcional)
# ------------------------------------------------------------------
plt.semilogy(qz, Rσ, label='σ-pol')
plt.semilogy(qz, Rπ, label='π-pol', ls='--')
plt.xlabel('$q_z$ (Å$^{-1}$)');  plt.ylabel('R')
plt.title('Reflectividad de película de 40 Å @ 833 eV')
plt.legend();  plt.tight_layout()
plt.show()