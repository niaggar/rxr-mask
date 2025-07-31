#!/usr/bin/env python3
"""Example of optimized workflow for parameter fitting using the new density profile methods.

This example demonstrates how to:
1. Build element data structure once at the beginning
2. Efficiently recalculate density profiles during optimization
3. Update only specific elements when needed
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Add the package to path
sys.path.insert(0, '/Users/niaggar/Developer/mitacs/rxr-mask')

from rxrmask.core.parameter import ParametersContainer
from rxrmask.core.compound import create_compound
from rxrmask.core.structure import Structure
from rxrmask.core.atom import Atom
from rxrmask.core.formfactor import FormFactorLocalDB


def create_test_structure():
    """Create a test structure for optimization workflow."""
    # Create parameter container
    params = ParametersContainer()
    
    # Create form factors and atoms
    fe_ff = FormFactorLocalDB(element="Fe", is_magnetic=False)
    o_ff = FormFactorLocalDB(element="O", is_magnetic=False)
    fe_atom = Atom(Z=26, name="Fe", ff=fe_ff)
    o_atom = Atom(Z=8, name="O", ff=o_ff)
    atoms = [fe_atom, o_atom]
    
    # Create compounds with some parameters marked for fitting
    fe_substrate = create_compound(
        parameters_container=params,
        name="Fe_substrate",
        formula="Fe:1",
        thickness=100.0,
        density=7.87,
        atoms=atoms,
        roughness=5.0
    )
    
    feo_layer = create_compound(
        parameters_container=params,
        name="FeO",
        formula="Fe:1,O:1",
        thickness=50.0,
        density=5.7,
        atoms=atoms,
        roughness=3.0
    )
    
    fe2o3_layer = create_compound(
        parameters_container=params,
        name="Fe2O3",
        formula="Fe:2,O:3",
        thickness=30.0,
        density=5.24,
        atoms=atoms,
        roughness=2.0
    )
    
    # Mark some parameters for fitting (example)
    feo_layer.thickness.fit = True
    fe2o3_layer.thickness.fit = True
    fe2o3_layer.roughness.fit = True
    
    # Create structure
    structure = Structure("Optimization_Example", 3)
    structure.add_compound(0, fe_substrate)
    structure.add_compound(1, feo_layer)
    structure.add_compound(2, fe2o3_layer)
    
    return structure, params


def demonstrate_optimized_workflow():
    """Demonstrate the optimized workflow for parameter fitting."""
    print("=== Optimized Density Profile Workflow Example ===\n")
    
    # Create test structure
    structure, params = create_test_structure()
    
    # STEP 1: Build element data once at the beginning
    print("1. Building element data structure (done once)...")
    element_data, layer_thickness_params, atoms = structure._create_element_data()
    print(f"   Elements found: {list(element_data.keys())}")
    print(f"   Layer thickness parameters: {len(layer_thickness_params)}")
    
    # STEP 2: Show initial density profiles
    print("\n2. Calculating initial density profiles...")
    z, dens, m_dens, _ = structure.get_density_profile_from_element_data(
        element_data, layer_thickness_params, atoms, step=1.0
    )
    print(f"   Spatial grid: {len(z)} points")
    
    # STEP 3: Simulate parameter updates during optimization
    print("\n3. Simulating optimization steps...")
    
    # Get initial fit parameters
    initial_fit_params = params.get_fit_vector()
    print(f"   Initial fit parameters: {initial_fit_params}")
    
    # Simulate some optimization steps
    optimization_steps = [
        [52.0, 28.0, 1.5],  # Step 1: modify thicknesses and roughness
        [55.0, 25.0, 2.2],  # Step 2: 
        [48.0, 32.0, 1.8],  # Step 3:
    ]
    
    performance_times = []
    
    for step_num, new_params in enumerate(optimization_steps, 1):
        print(f"\n   Optimization Step {step_num}:")
        print(f"     New parameters: {new_params}")
        
        # Update parameters (this is what the optimizer would do)
        params.set_fit_vector(new_params)
        
        # Measure performance of density profile recalculation
        start_time = time.time()
        
        # Recalculate all profiles (efficient because element_data has parameter references)
        z_new, dens_new, m_dens_new, _ = structure.get_density_profile_from_element_data(
            element_data, layer_thickness_params, atoms, step=1.0
        )
        
        # Calculate individual element profile (even more efficient)
        z_fe, profile_fe = structure.get_element_density_profile(
            "Fe", step=1.0, element_data=element_data, layer_thickness_params=layer_thickness_params
        )
        
        calc_time = time.time() - start_time
        performance_times.append(calc_time)
        
        print(f"     Calculation time: {calc_time:.4f} seconds")
        print(f"     Fe density range: {dens_new['Fe'].min():.3e} to {dens_new['Fe'].max():.3e}")
        print(f"     O density range: {dens_new['O'].min():.3e} to {dens_new['O'].max():.3e}")
    
    # STEP 4: Performance comparison
    print("\n4. Performance comparison...")
    
    # Time the traditional method (rebuilds everything)
    start_time = time.time()
    for _ in range(100):
        z_trad, dens_trad, m_dens_trad, atoms_trad = structure.get_density_profile(step=1.0)
    traditional_time = time.time() - start_time
    
    # Time the optimized method (reuses element_data)
    start_time = time.time()
    for _ in range(100):
        z_opt, dens_opt, m_dens_opt, _ = structure.get_density_profile_from_element_data(
            element_data, layer_thickness_params, atoms, step=1.0
        )
    optimized_time = time.time() - start_time
    
    # Time individual element calculation
    start_time = time.time()
    for _ in range(100):
        z_fe, profile_fe = structure.get_element_density_profile(
            "Fe", step=1.0, element_data=element_data, layer_thickness_params=layer_thickness_params
        )
    individual_time = time.time() - start_time
    
    print(f"   Traditional method (100x): {traditional_time:.4f} seconds")
    print(f"   Optimized method (100x): {optimized_time:.4f} seconds")
    print(f"   Individual element (100x): {individual_time:.4f} seconds")
    print(f"   Speedup (optimized): {traditional_time/optimized_time:.1f}x")
    print(f"   Speedup (individual): {traditional_time/individual_time:.1f}x")
    
    return z, dens, element_data, layer_thickness_params, atoms


def plot_optimization_results(z, dens, element_data, layer_thickness_params, atoms):
    """Plot the results of the optimization workflow."""
    print("\n5. Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Final density profiles
    ax1 = axes[0, 0]
    for element in dens:
        ax1.plot(z, dens[element], label=f"{element} density", linewidth=2)
    ax1.set_xlabel("Position (Å)")
    ax1.set_ylabel("Molar Density (mol/cm³)")
    ax1.set_title("Final Density Profiles")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Show parameter structure (conceptual)
    ax2 = axes[0, 1]
    element_names = list(element_data.keys())
    param_counts = [len([p for p in element_data[name]['density_params'] if p is not None]) 
                   for name in element_names]
    
    bars = ax2.bar(element_names, param_counts, alpha=0.7)
    ax2.set_ylabel("Number of Layer Parameters")
    ax2.set_title("Parameter Structure per Element")
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, param_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # Plot 3: Layer structure visualization
    ax3 = axes[1, 0]
    layer_thicknesses = [param.get() for param in layer_thickness_params]
    cumulative_pos = np.cumsum([0] + layer_thicknesses)
    
    for i, thickness in enumerate(layer_thicknesses):
        ax3.barh(0, thickness, left=cumulative_pos[i], alpha=0.6, 
                label=f"Layer {i+1} ({thickness:.1f} Å)")
    
    ax3.set_xlabel("Position (Å)")
    ax3.set_title("Layer Structure")
    ax3.legend()
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_yticks([])
    
    # Plot 4: Element distribution
    ax4 = axes[1, 1]
    z_sample = z[::10]  # Sample for clarity
    for element in dens:
        dens_sample = dens[element][::10]
        ax4.fill_between(z_sample, dens_sample, alpha=0.5, label=f"{element}")
    
    ax4.set_xlabel("Position (Å)")
    ax4.set_ylabel("Molar Density (mol/cm³)")
    ax4.set_title("Element Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/niaggar/Developer/mitacs/rxr-mask/optimization_workflow_example.png', dpi=150)
    print("   Plot saved as 'optimization_workflow_example.png'")
    
    return fig


def usage_examples():
    """Show practical usage examples."""
    print("\n=== Usage Examples ===\n")
    
    print("# Example 1: One-time calculation (simple)")
    print("z, dens, m_dens, atoms = structure.get_density_profile(step=0.1)")
    print()
    
    print("# Example 2: Optimized workflow (for fitting)")
    print("# Build element data once")
    print("element_data, layer_params, atoms = structure.get_element_data()")
    print()
    print("# During optimization loop:")
    print("for params in optimization_steps:")
    print("    params_container.set_fit_vector(params)")
    print("    z, dens, m_dens, _ = structure.get_density_profile_from_element_data(")
    print("        element_data, layer_params, atoms, step=0.1)")
    print("    # Use dens for objective function calculation")
    print()
    
    print("# Example 3: Calculate single element (most efficient)")
    print("z_fe, profile_fe = structure.get_element_density_profile(")
    print("    'Fe', step=0.1, element_data=element_data, layer_thickness_params=layer_params)")


if __name__ == "__main__":
    try:
        z, dens, element_data, layer_thickness_params, atoms = demonstrate_optimized_workflow()
        plot_optimization_results(z, dens, element_data, layer_thickness_params, atoms)
        usage_examples()
        
        print("\n✅ Optimized workflow demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
