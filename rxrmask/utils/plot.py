import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib import pyplot as plt

from rxrmask.core import AtomLayer


def plot_reflectivity(qz, R_phi, R_pi, energy_eV, model_name):
    """Plot X-ray reflectivity curves for both polarizations."""
    plt.figure(figsize=(8, 6))
    plt.semilogy(qz, R_phi, label=r"$\sigma$-pol")
    plt.semilogy(qz, R_pi, "--", label=r"$\pi$-pol")
    plt.xlabel(r"$q_z$ (Å$^{-1}$)")
    plt.ylabel(r"Reflectivity")
    plt.title(rf"Reflectivity for {model_name} at {energy_eV} eV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_energy_scan(e_pr, R_phi_pr, R_pi_pr, theta_deg, model_name):
    """Plot energy scan reflectivity at fixed angle."""
    plt.figure(figsize=(8, 6))
    plt.plot(e_pr, R_phi_pr, label=r"$\sigma$-pol")
    plt.plot(e_pr, R_pi_pr, "--", label=r"$\pi$-pol")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Reflectivity")
    plt.title(rf"Energy Scan for {model_name} at {theta_deg}°")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_density_profile(z, dens, figsize=(10, 4), title="Density Profile", x_move=0.0, min_x=0.0):
    """Plot atomic density profiles as function of depth."""
    plt.figure(figsize=figsize)
    for name, profile in dens.items():
        plt.plot(z + x_move, profile, "-", label=name)
    plt.xlabel("Depth $z$ (Å)")
    plt.title(title)
    plt.grid(True)
    plt.ylabel("Density $\\rho(z)$ (mol/cm³)")
    plt.ylim(bottom=0)
    plt.xlim(left=min_x)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_density_profile_atoms_layers(atoms_layers: list[AtomLayer], figsize=(10, 4), title="Density Profile", x_move=0.0, min_x=0.0):
    """Plot atomic density profiles as function of depth for AtomLayer objects."""
    plt.figure(figsize=figsize)
    for layer in atoms_layers:
        z = layer.z_deepness + x_move
        plt.plot(z, layer.molar_density, "-", label=layer.atom.name)
    plt.xlabel("Depth $z$ (Å)")
    plt.title(title)
    plt.grid(True)
    plt.ylabel("Density $\\rho(z)$ (mol/cm³)")
    plt.ylim(bottom=0)
    plt.xlim(left=min_x)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_slab_model(structure, figsize=(10, 4), cmap_name="tab10"):
    """Plot multilayer structure as colored rectangles showing layer composition."""
    thicknesses = [comp.thickness for comp in structure.compounds]
    densities = [comp.density for comp in structure.compounds]
    names = [comp.id for comp in structure.compounds]

    x0 = np.concatenate([[0], np.cumsum(thicknesses)])
    cmap = cm.get_cmap(cmap_name, structure.n_compounds)

    fig, ax = plt.subplots(figsize=figsize)
    for i, comp in enumerate(structure.compounds):
        rect = Rectangle(
            (x0[i], 0),
            width=thicknesses[i],
            height=densities[i],
            facecolor=cmap(i),
            edgecolor="k",
        )
        ax.add_patch(rect)
        ax.text(
            x0[i] + thicknesses[i] / 2,
            densities[i] / 2,
            names[i],
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            weight="bold",
        )

    for j, x in enumerate(x0):
        ax.axvline(x, linestyle="--", color="k")
        y_text = max(densities) * 1.02
        ax.text(x, y_text, f"[{j}]", ha="center", va="bottom", fontsize=10)

    ax.set_xlim(x0[0], x0[-1])
    ax.set_ylim(0, max(densities) * 1.1)
    ax.set_xlabel("Thickness (Å)")
    ax.set_ylabel("Density (g/cm³)")
    ax.set_title(f"{structure.name}")
    plt.tight_layout()
    plt.show()


def plot_formfactor_object(ff, title=None):
    """Plots f1 and f2 vs energy from a FormFactorLocalDB instance."""
    data = ff.get_all_formfactors()
    E = data[:, 0]
    f1 = data[:, 1]
    f2 = data[:, 2]

    plt.plot(E, f1, label="f1")
    plt.plot(E, f2, label="f2")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Form Factor")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
