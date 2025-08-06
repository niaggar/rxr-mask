# RXR-Mask

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](docs/)

RXR-Mask is a Python package for modeling and analyzing resonant X-ray reflectivity (RXR) from multilayer heterostructures. It provides a flexible framework for simulating X-ray reflectometry with support for magnetic materials, interface roughness, and multiple computational backends.

## Features

### **Core Capabilities**
- **Multilayer Structures**.
- **Resonant X-ray Reflectivity**.
- **Energy Scans**.
- **Magnetic Materials**.
- **Interface Roughness**.

### **Computational Backends**
- **Pythonreflectivity**: High-performance Cython-based calculations

## Installation

### Requirements
- Python ≥ 3.10
- NumPy, SciPy, Matplotlib, Cython (for compilation)
- Additional dependencies: see `requirements.txt`

### Install
```bash
git clone https://github.com/niaggar/rxr-mask.git
cd rxr-mask
pip install -e .
```

## Quick Start

### Basic Reflectivity Calculation

```python
import numpy as np
from rxrmask.core import Atom, Structure, FormFactorLocalDB, ParametersContainer
from rxrmask.core import create_compound
from rxrmask.backends import reflectivity_pr
from rxrmask.utils import plot_reflectivity

# Create atoms with form factors
sr_ff = FormFactorLocalDB(element="Sr", is_magnetic=False)
ti_ff = FormFactorLocalDB(element="Ti", is_magnetic=False) 
o_ff = FormFactorLocalDB(element="O", is_magnetic=False)

sr_atom = Atom(Z=38, name="Sr", ff=sr_ff)
ti_atom = Atom(Z=22, name="Ti", ff=ti_ff)
o_atom = Atom(Z=8, name="O", ff=o_ff)

# Create compound
parameters_container = ParametersContainer()
srtio3 = create_compound(
    parameters_container=parameters_container,
    name="SrTiO3",
    formula="Sr:1,Ti:1,O:3",
    thickness=50.0,
    density=5.12,
    atoms=[sr_atom, ti_atom, o_atom],
    roughness=2.0
)

# Build structure
structure = Structure(name="STO Film", n_compounds=1, params_container=parameters_container)
structure.add_compound(0, srtio3)
structure.validate_compounds()
structure.create_layers(step=0.1)

# Calculate reflectivity
E_eV = 600
theta = np.linspace(0.1, 89.1, num=1000)
qz = np.sin(theta * np.pi / 180) * (E_eV * 0.001013546143)

qz_calc, R_sigma, R_pi = reflectivity_pr(structure, qz, E_eV)

# Plot results
plot_reflectivity(qz_calc, R_sigma, R_pi, E_eV, "SrTiO3 Film")
```

### Energy Scan Analysis

```python
from rxrmask.backends import energy_scan_pr
from rxrmask.utils import plot_energy_scan

# Energy scan at fixed angle
energies = np.linspace(630, 670, num=200)
theta_deg = 15.0

e_calc, R_sigma, R_pi = energy_scan_pr(structure, energies, theta_deg)
plot_energy_scan(e_calc, R_sigma, R_pi, theta_deg, "SrTiO3 Energy Scan")
```

### Density Profile Visualization

```python
from rxrmask.utils import plot_density_profile, get_density_profile_from_element_data

# Calculate density profiles
z, density_profiles, magnetic_profiles, atoms = get_density_profile_from_element_data(
    structure.element_data, structure.atoms, structure.step
)

# Plot density profiles
plot_density_profile(z, density_profiles, title="SrTiO3 Density Profile")
```

## Project Structure

```
rxr-mask/
├── rxrmask/
│   ├── core/             # Core objects (Atoms, Compounds, Structures)
│   ├── backends/         # Computational engines (PythonReflectivity, UDKM)
│   ├── utils/            # Plotting and helper functions
│   └── materials/        # Local atomic form factor databases
├── pythonreflectivity/   # Cython backend implementation
├── docs/                 # Sphinx-based documentation
├── tests/                # Example notebooks and test cases
└── README.md
```

## Documentation

- **API Documentation**: Available in `docs/` directory
- **Tutorial Notebooks**:
  - `basic_compound.ipynb`: Basic usage examples
  - `group_elements.ipynb`: Complex heterostructures


## Credits

- Dr. Robert J. Green — Scientific supervision and mentorship
- QMax Group, University of Saskatchewan — Research environment
- MITACS — Internship support
- Martin Zwiebler — Author of the PythonReflectivity computational core

## Author

Developed by Nicolás Aguilera, undergraduate Physics student at Universidad del Valle (Colombia). This project was created as part of a research internship at the University of Saskatchewan (Canada).

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
