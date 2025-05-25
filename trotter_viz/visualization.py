"""
Interactive Visualization Tools for Trotterization

This module provides interactive plotting capabilities using Plotly and
Streamlit for exploring Trotterization methods.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .hamiltonians import QuantumHamiltonians, create_observables
from .trotterization import (
    create_lie_trotter_simulator,
    create_suzuki_trotter_simulator,
    compare_methods
)


class TrotterVisualizer:
    """Main visualization class for Trotterization analysis."""
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.color_palette = px.colors.qualitative.Set1
        self.default_figure_size = (800, 600)
    
    def plot_time_evolution(self, n_qubits: int = 4, hamiltonian_type: str = 'TFI',
                           max_time: float = 3.0, coupling: float = 1.0,
                           magnetic_field: float = 1.0, max_steps: int = 10) -> go.Figure:
        """
        Create an interactive plot showing time evolution under different Trotter methods.
        
        Args:
            n_qubits: Number of qubits in the system
            hamiltonian_type: Type of Hamiltonian ('TFI', 'XXZ', 'XY')
            max_time: Maximum evolution time
            coupling: Coupling strength parameter
            magnetic_field: Magnetic field strength
            max_steps: Maximum number of Trotter steps
            
        Returns:
            Plotly figure with time evolution comparison
        """
        # Create Hamiltonian
        if hamiltonian_type == 'TFI':
            hamiltonian = QuantumHamiltonians.transverse_field_ising(
                n_qubits, [magnetic_field] * n_qubits
            )
            title_suffix = f"TFI Chain (n={n_qubits}, g={magnetic_field:.1f})"
        elif hamiltonian_type == 'XXZ':
            hamiltonian = QuantumHamiltonians.xxz_heisenberg(n_qubits)
            title_suffix = f"XXZ Chain (n={n_qubits})"
        else:  # XY
            hamiltonian = QuantumHamiltonians.xy_model(
                n_qubits, [coupling] * (n_qubits-1), [magnetic_field] * n_qubits
            )
            title_suffix = f"XY Model (n={n_qubits}, J={coupling:.1f}, h={magnetic_field:.1f})"
        
        # Create observables
        observables = create_observables(n_qubits)
        
        # Time points
        times = np.linspace(0.1, max_time, 40)
        
        # Create simulators with corrected implementation
        simulators = {
            'Lie-Trotter': create_lie_trotter_simulator(),
            'Suzuki-2': create_suzuki_trotter_simulator(2),
            'Suzuki-4': create_suzuki_trotter_simulator(4)
        }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Correlation Function', 'Magnetization',
                'Method Comparison (t=1.5)', 'Circuit Complexity'
            )
        )
        
        # Plot time evolution for different step counts
        step_values = [1, max_steps//3, max_steps]
        
        for i, steps in enumerate(step_values):
            for j, (method_name, simulator) in enumerate(simulators.items()):
                # Generate circuits
                circuits = simulator.time_evolution(hamiltonian, times, steps)
                
                # Compute expectation values
                exp_vals = simulator.compute_expectation_values(
                    circuits, [observables['correlation'], observables['magnetization']]
                )
                
                # Plot correlation function
                color = self.color_palette[j % len(self.color_palette)]
                line_style = 'solid' if i == len(step_values)-1 else 'dash'
                
                fig.add_trace(
                    go.Scatter(
                        x=times, y=exp_vals[:, 0],
                        name=f'{method_name} (r={steps})',
                        legendgroup=f'{method_name} (r={steps})',
                        line=dict(color=color, dash=line_style),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Plot magnetization using the same legend group, but hide its legend entry
                fig.add_trace(
                    go.Scatter(
                        x=times, y=exp_vals[:, 1],
                        name=f'{method_name} (r={steps})',
                        legendgroup=f'{method_name} (r={steps})',
                        line=dict(color=color, dash=line_style),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Method comparison at fixed time
        comparison_time = 1.5
        steps_range = range(1, max_steps + 1)
        comparison_results = compare_methods(
            hamiltonian, comparison_time, observables['correlation'], steps_range
        )
        
        for i, (method_name, results) in enumerate(comparison_results.items()):
            color = self.color_palette[i % len(self.color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=list(steps_range), y=results,
                    name=method_name,
                    mode='lines+markers',
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Circuit complexity analysis
        complexity_data = self._analyze_circuit_complexity(hamiltonian, steps_range)
        
        for i, (method_name, depths) in enumerate(complexity_data.items()):
            color = self.color_palette[i % len(self.color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=list(steps_range), y=depths,
                    name=method_name,
                    mode='lines+markers',
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Trotter Steps", row=2, col=1)
        fig.update_xaxes(title_text="Trotter Steps", row=2, col=2)
        
        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Magnetization", row=1, col=2)
        fig.update_yaxes(title_text="Expectation Value", row=2, col=1)
        fig.update_yaxes(title_text="Circuit Depth", row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text=f"Trotterization Analysis: {title_suffix}",
            showlegend=True
        )
        
        return fig
    
    def plot_error_scaling(self, n_qubits: int = 3, hamiltonian_type: str = 'TFI',
                          time: float = 1.0, max_steps: int = 20) -> go.Figure:
        """
        Create a plot showing error scaling with Trotter steps.
        
        Args:
            n_qubits: Number of qubits
            hamiltonian_type: Type of Hamiltonian
            time: Evolution time
            max_steps: Maximum number of Trotter steps
            
        Returns:
            Plotly figure showing error scaling
        """
        # Create Hamiltonian
        if hamiltonian_type == 'TFI':
            hamiltonian = QuantumHamiltonians.transverse_field_ising(n_qubits, [1.0] * n_qubits)
        elif hamiltonian_type == 'XXZ':
            hamiltonian = QuantumHamiltonians.xxz_heisenberg(n_qubits)
        else:  # XY
            hamiltonian = QuantumHamiltonians.xy_model(
                n_qubits, [1.0] * (n_qubits-1), [1.0] * n_qubits
            )
        
        observables = create_observables(n_qubits)
        
        # High-precision reference using corrected implementation
        ref_simulator = create_lie_trotter_simulator()
        ref_circuit = ref_simulator.method.evolve(hamiltonian, time, 100)
        ref_value = ref_simulator.compute_expectation_values(
            [ref_circuit], [observables['correlation']]
        )[0, 0]
        
        # Test different methods with corrected implementation
        simulators = {
            'Lie-Trotter': create_lie_trotter_simulator(),
            'Suzuki-2': create_suzuki_trotter_simulator(2),
            'Suzuki-4': create_suzuki_trotter_simulator(4)
        }
        
        steps_range = range(1, max_steps + 1)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Error vs Trotter Steps', 'Theoretical Scaling')
        )
        
        for i, (method_name, simulator) in enumerate(simulators.items()):
            errors = []
            for steps in steps_range:
                circuit = simulator.method.evolve(hamiltonian, time, steps)
                exp_val = simulator.compute_expectation_values([circuit], [observables['correlation']])[0, 0]
                error = abs(exp_val - ref_value)
                errors.append(max(error, 1e-12))  # Avoid log(0)
            
            color = self.color_palette[i % len(self.color_palette)]
            
            # Actual errors
            fig.add_trace(
                go.Scatter(
                    x=list(steps_range), y=errors,
                    name=f'{method_name} Error',
                    mode='lines+markers',
                    line=dict(color=color)
                ),
                row=1, col=1
            )
            
            # Theoretical scaling
            if method_name == 'Lie-Trotter':
                theoretical = [1.0 / r for r in steps_range]
                scaling_label = 'O(1/r)'
            elif method_name == 'Suzuki-2':
                theoretical = [1.0 / (r**2) for r in steps_range]
                scaling_label = 'O(1/r²)'
            else:  # Suzuki-4
                theoretical = [1.0 / (r**4) for r in steps_range]
                scaling_label = 'O(1/r⁴)'
            
            # Normalize to match actual errors at r=5
            if len(errors) >= 5:
                norm_factor = errors[4] / theoretical[4]
                theoretical = [t * norm_factor for t in theoretical]
            
            fig.add_trace(
                go.Scatter(
                    x=list(steps_range), y=theoretical,
                    name=f'{method_name} {scaling_label}',
                    mode='lines',
                    line=dict(color=color, dash='dash'),
                    showlegend=True
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Trotter Steps", row=1, col=1)
        fig.update_xaxes(title_text="Trotter Steps", row=1, col=2)
        fig.update_yaxes(title_text="Error", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Theoretical Error", type="log", row=1, col=2)
        
        fig.update_layout(
            height=400,
            title_text="Trotter Error Scaling Analysis",
            showlegend=True
        )
        
        return fig
    
    def _analyze_circuit_complexity(self, hamiltonian, steps_range) -> Dict[str, List[int]]:
        """Analyze circuit complexity for different methods."""
        simulators = {
            'Lie-Trotter': create_lie_trotter_simulator(),
            'Suzuki-2': create_suzuki_trotter_simulator(2),
            'Suzuki-4': create_suzuki_trotter_simulator(4)
        }
        
        complexity_data = {}
        
        for method_name, simulator in simulators.items():
            depths = []
            for steps in steps_range:
                circuit = simulator.method.evolve(hamiltonian, 1.0, steps)
                depths.append(len(circuit))
            complexity_data[method_name] = depths
        
        return complexity_data
    
    def create_parameter_study(self, parameter_name: str, parameter_values: List[float],
                             n_qubits: int = 3, time: float = 2.0) -> go.Figure:
        """
        Create a parameter study visualization.
        
        Args:
            parameter_name: Name of parameter to vary ('coupling', 'field', 'time')
            parameter_values: List of parameter values to test
            n_qubits: Number of qubits
            time: Evolution time (ignored if parameter_name is 'time')
            
        Returns:
            Plotly figure showing parameter dependence
        """
        results = {
            'Lie-Trotter': [],
            'Suzuki-2': [],
            'Suzuki-4': []
        }
        
        simulators = {
            'Lie-Trotter': create_lie_trotter_simulator(),
            'Suzuki-2': create_suzuki_trotter_simulator(2),
            'Suzuki-4': create_suzuki_trotter_simulator(4)
        }
        
        for param_val in parameter_values:
            if parameter_name == 'coupling':
                hamiltonian = QuantumHamiltonians.transverse_field_ising(n_qubits, [param_val] * n_qubits)
            elif parameter_name == 'field':
                hamiltonian = QuantumHamiltonians.xy_model(
                    n_qubits, [1.0] * (n_qubits-1), [param_val] * n_qubits
                )
            elif parameter_name == 'time':
                hamiltonian = QuantumHamiltonians.transverse_field_ising(n_qubits, [1.0] * n_qubits)
                time = param_val
            else:
                raise ValueError(f"Unknown parameter: {parameter_name}")
            
            observables = create_observables(n_qubits)
            
            for method_name, simulator in simulators.items():
                circuit = simulator.method.evolve(hamiltonian, time, 5)
                exp_val = simulator.compute_expectation_values([circuit], [observables['correlation']])[0, 0]
                results[method_name].append(exp_val)
        
        fig = go.Figure()
        
        for i, (method_name, values) in enumerate(results.items()):
            color = self.color_palette[i % len(self.color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=parameter_values, y=values,
                    name=method_name,
                    mode='lines+markers',
                    line=dict(color=color)
                )
            )
        
        fig.update_layout(
            title=f"Parameter Study: {parameter_name.capitalize()}",
            xaxis_title=parameter_name.capitalize(),
            yaxis_title="Correlation Function",
            height=400
        )
        
        return fig


def create_dashboard_data(n_qubits: int = 4, max_time: float = 3.0) -> Dict:
    """
    Create comprehensive data for a dashboard visualization.
    
    Args:
        n_qubits: Number of qubits
        max_time: Maximum time for analysis
        
    Returns:
        Dictionary containing all data for dashboard plots
    """
    # Different Hamiltonians
    hamiltonians = {
        'TFI': QuantumHamiltonians.transverse_field_ising(n_qubits, [1.0] * n_qubits),
        'XXZ': QuantumHamiltonians.xxz_heisenberg(n_qubits),
        'XY': QuantumHamiltonians.xy_model(n_qubits, [1.0] * (n_qubits-1), [1.0] * n_qubits)
    }
    
    # Simulators with corrected implementation
    simulators = {
        'Lie-Trotter': create_lie_trotter_simulator(),
        'Suzuki-2': create_suzuki_trotter_simulator(2),
        'Suzuki-4': create_suzuki_trotter_simulator(4)
    }
    
    times = np.linspace(0.1, max_time, 30)
    
    dashboard_data = {
        'times': times,
        'hamiltonians': {},
        'methods': list(simulators.keys()),
        'n_qubits': n_qubits
    }
    
    for ham_name, hamiltonian in hamiltonians.items():
        observables = create_observables(n_qubits)
        ham_data = {}
        
        for method_name, simulator in simulators.items():
            circuits = simulator.time_evolution(hamiltonian, times, 5)
            exp_vals = simulator.compute_expectation_values(
                circuits, [observables['correlation'], observables['magnetization']]
            )
            
            ham_data[method_name] = {
                'correlation': exp_vals[:, 0],
                'magnetization': exp_vals[:, 1]
            }
        
        dashboard_data['hamiltonians'][ham_name] = ham_data
    
    return dashboard_data
