"""
Command-line demo script for Trotterization visualization.

This script demonstrates the key features of the Trotterization visualization
project without requiring a web interface.
"""

import sys
import os
# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from trotter_viz.visualization import TrotterVisualizer
from trotter_viz.hamiltonians import QuantumHamiltonians, create_observables
from trotter_viz.trotterization import (
    create_lie_trotter_simulator,
    create_suzuki_trotter_simulator,
    compare_methods
)


def run_basic_demo():
    """Run a basic demonstration of Trotterization methods."""
    print("👾 Trotterization Basic Demo")
    print("=" * 50)
    
    # Parameters
    n_qubits = 3
    coupling_strength = 1.0
    magnetic_field = 1.0
    max_time = 3.0
    
    print(f"System: {n_qubits} qubits")
    print(f"Hamiltonian: Transverse Field Ising")
    print(f"Coupling: {coupling_strength}, Field: {magnetic_field}")
    print(f"Evolution time: 0 to {max_time}")
    
    # Create Hamiltonian and observables
    hamiltonian = QuantumHamiltonians.transverse_field_ising(
        n_qubits, [magnetic_field] * n_qubits
    )
    observables = create_observables(n_qubits)
    
    # Time points
    times = np.linspace(0.1, max_time, 30)
    
    # Create simulators
    simulators = {
        'Lie-Trotter': create_lie_trotter_simulator(),
        'Suzuki-2': create_suzuki_trotter_simulator(2),
        'Suzuki-4': create_suzuki_trotter_simulator(4)
    }
    
    # Different Trotter step counts
    step_counts = [1, 3, 8]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Trotterization Methods Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green']
    
    # Plot 1: Correlation function vs time
    ax1 = axes[0, 0]
    for i, (method_name, simulator) in enumerate(simulators.items()):
        for j, steps in enumerate(step_counts):
            circuits = simulator.time_evolution(hamiltonian, times, steps)
            exp_vals = simulator.compute_expectation_values(
                circuits, [observables['correlation']]
            )
            
            line_style = '-' if j == len(step_counts)-1 else '--'
            alpha = 0.7 if j == 0 else 1.0
            
            ax1.plot(times, exp_vals[:, 0], color=colors[i], linestyle=line_style,
                    alpha=alpha, label=f'{method_name} (r={steps})' if j == len(step_counts)-1 else "")
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('ZZZ Correlation')
    ax1.set_title('Correlation Function Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Magnetization vs time
    ax2 = axes[0, 1]
    for i, (method_name, simulator) in enumerate(simulators.items()):
        circuits = simulator.time_evolution(hamiltonian, times, 5)  # Fixed steps
        exp_vals = simulator.compute_expectation_values(
            circuits, [observables['magnetization']]
        )
        ax2.plot(times, exp_vals[:, 0], color=colors[i], label=method_name)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Magnetization')
    ax2.set_title('Magnetization Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error scaling
    ax3 = axes[1, 0]
    steps_range = range(1, 16)
    fixed_time = 1.5
    
    # High-precision reference
    ref_simulator = create_lie_trotter_simulator()
    ref_circuit = ref_simulator.method.evolve(hamiltonian, fixed_time, 50)
    ref_value = ref_simulator.compute_expectation_values(
        [ref_circuit], [observables['correlation']]
    )[0, 0]
    
    for i, (method_name, simulator) in enumerate(simulators.items()):
        errors = []
        for steps in steps_range:
            circuit = simulator.method.evolve(hamiltonian, fixed_time, steps)
            exp_val = simulator.compute_expectation_values(
                [circuit], [observables['correlation']]
            )[0, 0]
            error = abs(exp_val - ref_value)
            errors.append(max(error, 1e-12))
        
        ax3.semilogy(list(steps_range), errors, 'o-', color=colors[i], 
                    label=method_name)
    
    ax3.set_xlabel('Trotter Steps')
    ax3.set_ylabel('Error (log scale)')
    ax3.set_title('Error vs Trotter Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Circuit complexity
    ax4 = axes[1, 1]
    complexity_steps = range(1, 11)
    
    for i, (method_name, simulator) in enumerate(simulators.items()):
        depths = []
        for steps in complexity_steps:
            circuit = simulator.method.evolve(hamiltonian, 1.0, steps)
            depths.append(len(circuit))
        
        ax4.plot(list(complexity_steps), depths, 'o-', color=colors[i], 
                label=method_name)
    
    ax4.set_xlabel('Trotter Steps')
    ax4.set_ylabel('Circuit Depth')
    ax4.set_title('Circuit Complexity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Demo Results:")
    print("• Solid lines show high Trotter step count (r=8)")
    print("• Dashed lines show low Trotter step count (r=1)")
    print("• Higher-order methods converge faster with fewer steps")
    print("• Trade-off: accuracy vs. circuit complexity")


def run_comparison_study():
    """Run a detailed comparison study."""
    print("\n🔬 Detailed Comparison Study")
    print("=" * 50)
    
    # Test different Hamiltonians
    hamiltonians = {
        'TFI': QuantumHamiltonians.transverse_field_ising(3, [1.0, 1.0, 1.0]),
        'XXZ': QuantumHamiltonians.xxz_heisenberg(3),
        'XY': QuantumHamiltonians.xy_model(3, [1.0, 1.0], [0.5, 0.5, 0.5])
    }
    
    time = 2.0
    steps_range = range(1, 11)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['blue', 'red', 'green']
    
    for i, (ham_name, hamiltonian) in enumerate(hamiltonians.items()):
        ax = axes[i]
        observables = create_observables(3)
        
        # Compare methods
        comparison_results = compare_methods(
            hamiltonian, time, observables['correlation'], steps_range
        )
        
        for j, (method_name, results) in enumerate(comparison_results.items()):
            ax.plot(list(steps_range), results, 'o-', color=colors[j], 
                   label=method_name)
        
        ax.set_xlabel('Trotter Steps')
        ax.set_ylabel('Correlation Function')
        ax.set_title(f'{ham_name} Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Hamiltonian Model Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\n🎯 Key Observations:")
    print("• Different Hamiltonians show varying sensitivity to Trotter approximation")
    print("• TFI model: Strong dependence on evolution method")
    print("• XXZ model: More stable across different methods")
    print("• XY model: Intermediate behavior")


def print_theoretical_background():
    """Print educational information about Trotterization."""
    print("\n📚 Theoretical Background")
    print("=" * 50)
    
    print("""
Trotterization Methods for Quantum Simulation:

1. LIE-TROTTER FORMULA (First Order):
   exp(-i(H₁ + H₂ + ...)t) ≈ (exp(-iH₁t/r)exp(-iH₂t/r)...)ʳ
   
   • Error scaling: O(t²/r)
   • Circuit depth: Linear in r
   • Best for: Quick approximations, limited resources

2. SUZUKI-TROTTER (Second Order):
   Uses symmetric decomposition for better accuracy
   
   • Error scaling: O(t³/r²)  
   • Circuit depth: ~2x Lie-Trotter
   • Best for: Balanced accuracy/cost applications

3. HIGHER-ORDER SUZUKI (4th, 6th, ...):
   Recursive construction for superior accuracy
   
   • Error scaling: O(t^(p+1)/r^p) for p-th order
   • Circuit depth: Exponential growth with order
   • Best for: High-precision requirements

APPLICATIONS:
• Quantum Chemistry: Molecular dynamics simulation
• Condensed Matter: Phase transition studies  
• Quantum Machine Learning: Algorithm implementation
• Optimization: QAOA and related algorithms

TRADE-OFFS:
✓ Higher order → Better accuracy
✗ Higher order → More quantum gates
✓ More steps → Better approximation  
✗ More steps → Longer computation time
""")


def main():
    """Main demo function."""
    print("⚛️  TROTTERIZATION VISUALIZATION PROJECT")
    print("=" * 60)
    print("Interactive exploration of quantum Hamiltonian simulation")
    print("=" * 60)
    
    try:
        # Run basic demo
        run_basic_demo()
        
        # Run comparison study
        run_comparison_study()
        
        # Print educational content
        print_theoretical_background()
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("• Run 'streamlit run dashboard.py' for interactive web interface")
        print("• Explore different parameters and Hamiltonians")
        print("• Study error scaling for your specific use case")
        
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("Install with: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("Check that all dependencies are correctly installed")


if __name__ == "__main__":
    main()
