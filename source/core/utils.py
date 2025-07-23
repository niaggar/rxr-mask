import numpy as np
from scipy.special import erf


def step_function(z, zi):
    return np.where(z >= zi, 2.0, 0.0)

def density_profile(thicknesses, densities, roughness, step=0.1):
    pos = np.cumsum(thicknesses)

    max_roughness = max(roughness.values(), key=lambda x: x[-1])[-1]
    z = np.arange(0, pos[-1] + max_roughness*1.5, step)
    dens = {}

    for name, rho in densities.items():
        profile = np.full_like(z, rho[0], dtype=float)

        for i in range(len(thicknesses)):
            drho = rho[i+1] - rho[i]
            rough = roughness[name][i]

            if rough == 0:
                profile += 0.5 * (drho) * step_function(z, pos[i])
            else:
                profile += (drho/2) * (erf((z-pos[i])/(rough/np.sqrt(2))) + 1)

        dens[name] = profile
    return z, dens

def z_density_profile(z_eval, thicknesses, densities, roughness):
    pos = np.cumsum(thicknesses)

    z = np.array(z_eval)
    dens = {}

    for name, rho in densities.items():
        profile = np.full_like(z, rho[0], dtype=float)

        for i in range(len(thicknesses)):
            drho = rho[i+1] - rho[i]
            rough = roughness[name][i]

            if rough == 0:
                profile += 0.5 * (drho) * step_function(z, pos[i])
            else:
                profile += (drho/2) * (erf((z-pos[i])/(rough/np.sqrt(2))) + 1)

        dens[name] = profile
    return z, dens
