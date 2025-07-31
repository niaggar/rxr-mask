#!/usr/bin/env python3
"""Example demonstrating optimized layer creation and updating workflow.

This example shows how to:
1. Create layers efficiently using pre-computed element data
2. Update layer densities when parameters change
3. Use the optimized workflow for parameter fitting scenarios
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
    """Create a test structure for layer optimization demonstration."""
    # Create parameter container
    params = ParametersContainer()
    
    # Create form factors and atoms
    fe_ff = FormFactorLocalDB(element="Fe", is_magnetic=False)
    o_ff = FormFactorLocalDB(element="O", is_magnetic=False)
    fe_atom = Atom(Z=26, name="Fe", ff=fe_ff)
    o_atom = Atom(Z=8, name="O", ff=o_ff)
    atoms = [fe_atom, o_atom]
    
    # Create compounds
    fe_substrate = create_compound(
        parameters_container=params,
        name="Fe_substrate",
        formula="Fe:1",
        thickness=50.0,  # Smaller for demonstration
        density=7.87,
        atoms=atoms,
        roughness=2.0
    )
    
    feo_layer = create_compound(
        parameters_container=params,
        name="FeO",
        formula="Fe:1,O:1",
        thickness=20.0,
        density=5.7,
        atoms=atoms,
        roughness=1.5
    )
    
    fe2o3_layer = create_compound(
        parameters_container=params,
        name="Fe2O3",
        formula="Fe:2,O:3",
        thickness=15.0,
        density=5.24,
        atoms=atoms,
        roughness=1.0
    )
    
    # Mark some parameters for fitting
    feo_layer.thickness.fit = True
    fe2o3_layer.thickness.fit = True
    
    # Create structure
    structure = Structure("Layer_Optimization_Example", 3)
    structure.add_compound(0, fe_substrate)
    structure.add_compound(1, feo_layer)
    structure.add_compound(2, fe2o3_layer)
    
    return structure, params


def demonstrate_layer_workflows():
    """Demonstrate different layer creation and updating workflows."""
    print("=== Layer Creation and Updating Workflows ===\n")
    
    structure, params = create_test_structure()
    step_size = 2.0  # Larger step for demonstration
    
    # WORKFLOW 1: Traditional approach (create layers directly)
    print("1. Traditional Layer Creation Workflow:")
    start_time = time.time()
    structure.create_layers(step=step_size)
    traditional_time = time.time() - start_time
    
    print(f"   Created {structure.n_layers} layers in {traditional_time:.4f} seconds")
    summary = structure.get_layers_summary()
    print(f"   Total thickness: {summary['total_thickness']:.1f} Å")
    print(f"   Elements: {summary['elements']}")
    
    # WORKFLOW 2: Optimized approach (prepare element data once, reuse for layers)
    print("\n2. Optimized Layer Creation Workflow:")
    
    # Step 2a: Build element data once
    print("   2a. Building element data (done once)...")
    start_time = time.time()
    element_data, layer_thickness_params, atoms = structure._create_element_data()
    element_data_time = time.time() - start_time
    print(f"       Element data built in {element_data_time:.4f} seconds")
    
    # Step 2b: Create layers using pre-built element data
    print("   2b. Creating layers with optimized method...")
    start_time = time.time()
    layers, layer_params_container = structure.create_layers_optimized(
        step=step_size, 
        element_data=element_data, 
        layer_thickness_params=layer_thickness_params, 
        atoms=atoms
    )
    optimized_creation_time = time.time() - start_time
    
    print(f"       Created {len(layers)} layers in {optimized_creation_time:.4f} seconds")
    print(f"       Layer parameters container has {len(layer_params_container.parameters)} parameters")
    
    # WORKFLOW 3: Parameter updates and layer recalculation
    print("\n3. Parameter Update and Layer Recalculation:")
    
    # Get initial fit parameters
    initial_params = params.get_fit_vector()
    print(f"   Initial fit parameters: {initial_params}")
    
    # Simulate parameter updates during optimization
    optimization_steps = [
        [22.0, 18.0],  # New thicknesses
        [25.0, 12.0],
        [19.0, 20.0],
    ]
    
    update_times = []
    
    for step_num, new_params in enumerate(optimization_steps, 1):
        print(f"\n   Step {step_num}: Updating parameters to {new_params}")
        
        # Update parameters
        params.set_fit_vector(new_params)
        
        # Method A: Update layers using the efficient update method
        start_time = time.time()
        structure.update_layers_from_density_profiles(
            element_data, layer_thickness_params, atoms, step=step_size
        )
        update_time = time.time() - start_time
        update_times.append(update_time)
        
        print(f"       Layers updated in {update_time:.4f} seconds")
        
        # Show layer summary
        summary = structure.get_layers_summary()
        print(f"       New total thickness: {summary['total_thickness']:.1f} Å")
        print(f"       Number of layers: {summary['n_layers']}")
    
    # WORKFLOW 4: Performance comparison
    print("\n4. Performance Comparison:")
    
    # Compare traditional vs optimized for multiple updates
    n_iterations = 50
    
    # Traditional method (recreate everything each time)
    print(f"   Testing traditional method ({n_iterations} iterations)...")
    start_time = time.time()
    for i in range(n_iterations):
        # Simulate parameter change
        test_params = [20.0 + i*0.1, 15.0 + i*0.1]
        params.set_fit_vector(test_params)
        structure.create_layers(step=step_size)
    traditional_total_time = time.time() - start_time
    
    # Optimized method (reuse element data)
    print(f"   Testing optimized method ({n_iterations} iterations)...")
    element_data, layer_thickness_params, atoms = structure._create_element_data()  # Rebuild once
    start_time = time.time()
    for i in range(n_iterations):
        # Simulate parameter change
        test_params = [20.0 + i*0.1, 15.0 + i*0.1]
        params.set_fit_vector(test_params)
        structure.update_layers_from_density_profiles(
            element_data, layer_thickness_params, atoms, step=step_size
        )
    optimized_total_time = time.time() - start_time
    
    print(f"   Traditional method total: {traditional_total_time:.4f} seconds")
    print(f"   Optimized method total: {optimized_total_time:.4f} seconds")
    print(f"   Speedup: {traditional_total_time/optimized_total_time:.1f}x")
    
    return structure, element_data, layer_thickness_params, atoms


def visualize_layer_structure(structure, element_data, layer_thickness_params, atoms):
    """Visualize the layer structure and density profiles."""
    print("\n5. Visualizing Layer Structure:")
    
    # Get current density profiles
    z, dens, m_dens, _ = structure.get_density_profile(step=1.0)
    
    # Get layer summary
    summary = structure.get_layers_summary()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Density profiles
    ax1 = axes[0, 0]
    for element in dens:
        ax1.plot(z, dens[element], label=f"{element} density", linewidth=2)
    ax1.set_xlabel("Position (Å)")
    ax1.set_ylabel("Molar Density (mol/cm³)")
    ax1.set_title("Continuous Density Profiles")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Layer structure
    ax2 = axes[0, 1]
    cumulative_pos = 0
    layer_positions = []
    layer_widths = []
    
    for i, layer_info in enumerate(summary['layer_info']):
        layer_widths.append(layer_info['thickness'])
        layer_positions.append(cumulative_pos)
        cumulative_pos += layer_info['thickness']
    
    # Create stacked bar chart for layers
    import matplotlib.cm as cm
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(summary['layer_info'])))
    for i, (pos, width) in enumerate(zip(layer_positions, layer_widths)):
        ax2.barh(0, width, left=pos, alpha=0.7, color=colors[i], 
                label=f"Layer {i+1} ({width:.1f} Å)")
    
    ax2.set_xlabel("Position (Å)")
    ax2.set_title("Discretized Layer Structure")
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    # Only show legend if not too many layers
    if len(summary['layer_info']) <= 10:
        ax2.legend()
    
    # Plot 3: Layer density values
    ax3 = axes[1, 0]
    layer_indices = range(len(summary['layer_info']))
    
    for element in summary['elements']:
        element_densities = []
        for layer_info in summary['layer_info']:
            if element in layer_info['elements']:
                element_densities.append(layer_info['elements'][element]['molar_density'])
            else:
                element_densities.append(0.0)
        
        ax3.plot(layer_indices, element_densities, 'o-', label=f"{element}", markersize=4)
    
    ax3.set_xlabel("Layer Index")
    ax3.set_ylabel("Molar Density (mol/cm³)")
    ax3.set_title("Layer-by-Layer Densities")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter structure visualization
    ax4 = axes[1, 1]
    
    # Count parameters by type
    param_types = ['Compound Parameters', 'Layer Parameters']
    param_counts = [len(structure.get_element_data()[1]), len(summary['layer_info']) * len(summary['elements']) * 2]
    
    bars = ax4.bar(param_types, param_counts, alpha=0.7, color=['lightblue', 'lightcoral'])
    ax4.set_ylabel("Number of Parameters")
    ax4.set_title("Parameter Structure")
    
    # Add value labels on bars
    for bar, count in zip(bars, param_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_counts)*0.01,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/niaggar/Developer/mitacs/rxr-mask/layer_optimization_example.png', dpi=150)
    print("   Plot saved as 'layer_optimization_example.png'")
    
    return fig


def usage_examples():
    """Show practical usage examples for different scenarios."""
    print("\n=== Usage Examples ===\n")
    
    print("# Scenario 1: Simple layer creation (one-time calculation)")
    print("structure.create_layers(step=0.1)")
    print("summary = structure.get_layers_summary()")
    print()
    
    print("# Scenario 2: Optimized workflow for fitting/optimization")
    print("# Step 1: Build element data once")
    print("element_data, layer_params, atoms = structure.get_element_data()")
    print()
    print("# Step 2: Create layers with parameter management")
    print("layers, layer_params_container = structure.create_layers_optimized(")
    print("    step=0.1, element_data=element_data, layer_thickness_params=layer_params, atoms=atoms)")
    print()
    print("# Step 3: During optimization loop")
    print("for new_params in optimization_steps:")
    print("    params_container.set_fit_vector(new_params)")
    print("    structure.update_layers_from_density_profiles(")
    print("        element_data, layer_params, atoms, step=0.1)")
    print("    # Use structure.layers for reflectivity calculations")
    print()
    
    print("# Scenario 3: Real-time layer analysis")
    print("# Monitor layer structure during optimization")
    print("summary = structure.get_layers_summary()")
    print("print(f'Total thickness: {summary[\"total_thickness\"]:.2f} Å')")
    print("print(f'Number of layers: {summary[\"n_layers\"]}')")


if __name__ == "__main__":
    try:
        structure, element_data, layer_thickness_params, atoms = demonstrate_layer_workflows()
        visualize_layer_structure(structure, element_data, layer_thickness_params, atoms)
        usage_examples()
        
        print("\n✅ Layer optimization demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
