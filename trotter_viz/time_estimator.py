from typing import Dict

class TimeEstimator:
    """Estimates computation time for different plot types based on parameters."""
    
    def __init__(self):
        """Initialize with baseline timing measurements."""
        # Baseline times in seconds for reference calculations
        # Updated based on comprehensive performance measurements
        self.base_times = {
            'circuit_evolution': 0.002,  # per circuit per time point
            'expectation_value': 0.001,  # per expectation value calculation
            'method_comparison': 0.0005, # per comparison point
            'parameter_study': 0.002,    # per parameter value
            'error_scaling': 0.003       # per error calculation
        }
        
        # Complexity multipliers for different system sizes
        # Updated based on new measurements
        self.complexity_factors = {
            2: 1.0,    # 2 qubits (baseline)
            3: 6.0,    # 3 qubits (20.5s TFI / ~3.4s baseline = 6x)
            4: 8.0,    # 4 qubits (12.4s TFI / ~1.5s baseline = 8x)
            5: 12.0,   # 5 qubits (extrapolated)
            6: 16.0    # 6 qubits (extrapolated)
        }
          # Hamiltonian-specific complexity factors
        # Based on actual measurements showing dramatic differences
        self.hamiltonian_factors = {
            'TFI': 1.0,    # Transverse Field Ising (baseline, fastest)
            'XXZ': 4.0,    # XXZ Heisenberg (45.9s vs 20.5s TFI = ~2.2x, but accounting for 3 vs 4 qubits)
            'XY': 6.0      # XY Model (95.3s vs 12.4s TFI = ~7.7x for same 4 qubits)
        }
    
    def estimate_time_evolution_plot(self, n_qubits: int, max_time: float, 
                                   max_steps: int, hamiltonian_type: str = 'TFI', time_points: int = 40) -> Dict[str, float]:
        """
        Estimate time for generating time evolution plots.
        
        Args:
            n_qubits: Number of qubits
            max_time: Maximum evolution time
            max_steps: Maximum Trotter steps
            hamiltonian_type: Type of Hamiltonian ('TFI', 'XXZ', 'XY')
            time_points: Number of time points to calculate
            
        Returns:
            Dictionary with time estimates for different components
        """
        complexity_factor = self.complexity_factors.get(n_qubits, 20.0)
        hamiltonian_factor = self.hamiltonian_factors.get(hamiltonian_type, 1.0)
        
        # Steps factor - more Trotter steps increase computation time
        steps_factor = 1.0 + (max_steps - 5) * 0.05  # Small increase per additional step
        
        # Main time evolution plots - updated based on actual measurements
        # Base time accounts for qubit complexity, Hamiltonian type, and steps
        base_plot_time = 1.5 * complexity_factor * hamiltonian_factor * steps_factor
        
        # Method and analysis overhead
        method_overhead = 0.1 * base_plot_time
        complexity_analysis_time = 0.1 * base_plot_time
        
        total_time = base_plot_time + method_overhead + complexity_analysis_time
        
        return {
            'main_evolution': base_plot_time,
            'method_comparison': method_overhead,
            'complexity_analysis': complexity_analysis_time,
            'total': total_time,
            'display_total': max(total_time, 1.0)  # Minimum 1 second for display
        }
    def estimate_error_scaling_plot(self, n_qubits: int, max_steps: int) -> Dict[str, float]:
        """
        Estimate time for error scaling analysis.
        
        Args:
            n_qubits: Number of qubits
            max_steps: Maximum number of steps to analyze
            
        Returns:
            Dictionary with time estimates
        """
        complexity_factor = self.complexity_factors.get(n_qubits, 3.5)
        
        # Simplified estimation based on actual performance
        base_time = 1.5 * complexity_factor  # Base time for error scaling
        step_overhead = max_steps * 0.02 * complexity_factor  # Small per-step overhead
        
        total_time = base_time + step_overhead
        
        return {
            'reference_calculation': base_time * 0.3,
            'error_calculations': base_time * 0.7 + step_overhead,
            'total': total_time,
            'display_total': max(total_time, 0.5)
        }
    def estimate_parameter_study(self, parameter_values: int, n_qubits: int) -> Dict[str, float]:
        """
        Estimate time for parameter study plots.
        
        Args:
            parameter_values: Number of parameter values to test
            n_qubits: Number of qubits
            
        Returns:
            Dictionary with time estimates
        """
        complexity_factor = self.complexity_factors.get(n_qubits, 3.5)
        
        # Simplified estimation: base time + small per-parameter overhead
        base_time = 1.0 * complexity_factor
        parameter_overhead = parameter_values * 0.1 * complexity_factor
        
        total_time = base_time + parameter_overhead
        
        return {
            'parameter_sweep': total_time,
            'total': total_time,
            'display_total': max(total_time, 0.3)
        }
    
    def format_time_estimate(self, time_seconds: float) -> str:
        """
        Format time estimate for user display.
        
        Args:
            time_seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if time_seconds < 1:
            return f"~{time_seconds:.1f}s"
        elif time_seconds < 60:
            return f"~{int(time_seconds)}s"
        elif time_seconds < 3600:
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            return f"~{minutes}m {seconds}s"
        else:
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            return f"~{hours}h {minutes}m"