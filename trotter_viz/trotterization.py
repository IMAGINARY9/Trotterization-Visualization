"""
Trotterization Methods for Quantum Hamiltonian Simulation

This module implements various Trotterization algorithms for approximating
quantum time evolution operators.
"""

import cirq
import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod

# Import quantum simulation core functions
from .quantum_simulation_core import (
    create_exponential_circuits, 
    compute_expectation_values,
    _create_pauli_exponential,
    lie_trotter,
    second_order_suzuki_trotter
)


class TrotterMethod(ABC):
    """Abstract base class for Trotterization methods."""
    
    @abstractmethod
    def evolve(self, hamiltonian: cirq.PauliSum, time: float, steps: int) -> cirq.Circuit:
        """
        Apply the Trotterization method to evolve under the given Hamiltonian.
        
        Args:
            hamiltonian: The Hamiltonian to evolve under
            time: Total evolution time
            steps: Number of Trotter steps
            
        Returns:
            Quantum circuit implementing the time evolution
        """
        pass


class LieTrotter(TrotterMethod):
    """
    First-order Lie-Trotter decomposition.
    
    Approximates exp(-iHt) ≈ (exp(-iH₁t/r)exp(-iH₂t/r)...exp(-iHₘt/r))ʳ
    where H = H₁ + H₂ + ... + Hₘ and r is the number of steps.
    
    Error scaling: O(t²/r)
    """
    
    def evolve(self, hamiltonian: cirq.PauliSum, time: float, steps: int) -> cirq.Circuit:
        """Apply first-order Lie-Trotter decomposition."""
        circuit = cirq.Circuit()
        dt = time / steps
        
        for _ in range(steps):
            circuit += self._single_trotter_step(hamiltonian, dt)
            
        return circuit
    def _single_trotter_step(self, hamiltonian: cirq.PauliSum, dt: float) -> cirq.Circuit:
        """Create a single Trotter step."""
        step_circuit = cirq.Circuit()
        
        for term in hamiltonian:
            # Each term is evolved separately using our helper method
            step_circuit += self._create_pauli_evolution(term, dt)
            
        return step_circuit
    
    def _create_pauli_evolution(self, pauli_string, angle: float) -> cirq.Circuit:
        """
        Create a circuit for evolving under a single Pauli string.
        
        Uses proper tensor product evolution with CNOT ladder decomposition.
        """
        # Extract coefficient
        if hasattr(pauli_string, 'coefficient'):
            coeff = float(pauli_string.coefficient.real)
        else:
            coeff = 1.0
        
        # Create scaled Pauli string with the correct coefficient for the evolution
        scaled_pauli = pauli_string * (angle / coeff) if coeff != 0 else pauli_string * angle
        
        return _create_pauli_exponential(scaled_pauli)


class SuzukiTrotter(TrotterMethod):
    """
    Higher-order Suzuki-Trotter decomposition.
    
    Implements symmetric (second-order) and higher-order decompositions
    for improved accuracy.
    """
    
    def __init__(self, order: int = 2):
        """
        Initialize Suzuki-Trotter method.
        
        Args:
            order: Order of the decomposition (2, 4, 6, ...)
        """
        if order < 2 or order % 2 != 0:
            raise ValueError("Order must be an even integer ≥ 2")
        self.order = order
    
    def evolve(self, hamiltonian: cirq.PauliSum, time: float, steps: int) -> cirq.Circuit:
        """Apply Suzuki-Trotter decomposition."""
        if self.order == 2:
            return self._second_order_suzuki(hamiltonian, time, steps)
        else:
            return self._higher_order_suzuki(hamiltonian, time, steps)
    
    def _second_order_suzuki(self, hamiltonian: cirq.PauliSum, time: float, steps: int) -> cirq.Circuit:
        """
        Second-order symmetric Suzuki-Trotter.
        
        Uses the symmetric decomposition:
        exp(-iHt/r) ≈ exp(-iH₁t/2r)exp(-iH₂t/2r)...exp(-iHₘt/r)...exp(-iH₂t/2r)exp(-iH₁t/2r)
        """
        circuit = cirq.Circuit()
        dt = time / steps
        
        for _ in range(steps):
            # Forward half-steps
            circuit += self._half_step_forward(hamiltonian, dt)
            # Backward half-steps
            circuit += self._half_step_backward(hamiltonian, dt)
            
        return circuit
    
    def _half_step_forward(self, hamiltonian: cirq.PauliSum, dt: float) -> cirq.Circuit:
        """Forward half-step for symmetric decomposition."""
        step_circuit = cirq.Circuit()
        terms = list(hamiltonian)
        
        # First half of terms with dt/2
        for i in range(len(terms) // 2):
            step_circuit += self._create_pauli_evolution(terms[i], dt/2)
        
        # Middle term(s) with full dt
        if len(terms) % 2 == 1:
            middle_idx = len(terms) // 2
            step_circuit += self._create_pauli_evolution(terms[middle_idx], dt)
        
        return step_circuit
    
    def _half_step_backward(self, hamiltonian: cirq.PauliSum, dt: float) -> cirq.Circuit:
        """Backward half-step for symmetric decomposition."""
        step_circuit = cirq.Circuit()
        terms = list(hamiltonian)
        
        # Remaining terms with dt/2 in reverse order
        start_idx = (len(terms) + 1) // 2
        for i in reversed(range(start_idx, len(terms))):
            step_circuit += self._create_pauli_evolution(terms[i], dt/2)
        return step_circuit
    
    def _create_pauli_evolution(self, pauli_string, angle: float) -> cirq.Circuit:
        """
        Create a circuit for evolving under a single Pauli string.
        
        Uses proper tensor product evolution with CNOT ladder decomposition.
        """
        # Extract coefficient
        if hasattr(pauli_string, 'coefficient'):
            coeff = float(pauli_string.coefficient.real)
        else:
            coeff = 1.0
        
        # Create scaled Pauli string with the correct coefficient for the evolution
        scaled_pauli = pauli_string * (angle / coeff) if coeff != 0 else pauli_string * angle
        
        return _create_pauli_exponential(scaled_pauli)
    
    def _higher_order_suzuki(self, hamiltonian: cirq.PauliSum, time: float, steps: int) -> cirq.Circuit:
        """
        Higher-order Suzuki decomposition using recursive construction.
        
        Uses the recursive formula for higher-order decompositions.
        """
        if self.order == 2:
            return self._second_order_suzuki(hamiltonian, time, steps)
        
        # Recursive coefficients for higher orders
        p = 1.0 / (4.0 - 4.0**(1.0/(self.order - 1)))
        
        circuit = cirq.Circuit()
        
        # Recursive construction
        sub_method = SuzukiTrotter(self.order - 2)
        sub_circuit = sub_method.evolve(hamiltonian, p * time, steps)
        
        # S_k = S_{k-2}(p*t) ∘ S_{k-2}(p*t) ∘ S_{k-2}((1-4p)*t) ∘ S_{k-2}(p*t) ∘ S_{k-2}(p*t)
        circuit += sub_circuit
        circuit += sub_circuit
        circuit += sub_method.evolve(hamiltonian, (1 - 4*p) * time, steps)
        circuit += sub_circuit
        circuit += sub_circuit
        
        return circuit


class TrotterSimulator:
    """
    Main simulator class that combines Hamiltonians with Trotterization methods.
    """
    
    def __init__(self, method: TrotterMethod):
        """
        Initialize the simulator.
        
        Args:
            method: Trotterization method to use
        """
        self.method = method
    
    def time_evolution(self, hamiltonian: cirq.PauliSum, times: np.ndarray, 
                      steps: int) -> List[cirq.Circuit]:
        """
        Generate time evolution circuits for multiple time points.
        
        Args:
            hamiltonian: Hamiltonian to evolve under
            times: Array of time points
            steps: Number of Trotter steps per time point
            
        Returns:
            List of circuits for each time point
        """
        circuits = []
        for t in times:
            circuit = self.method.evolve(hamiltonian, float(t), steps)
            circuits.append(circuit)
        return circuits
    
    def compute_expectation_values(self, circuits: List[cirq.Circuit], 
                                 observables: List[cirq.PauliSum]) -> np.ndarray:
        """
        Compute expectation values for given circuits and observables.
        
        Args:
            circuits: List of quantum circuits
            observables: List of observable operators
              Returns:
            Array of expectation values with shape (n_circuits, n_observables)
        """
        from .quantum_simulation_core import compute_expectation_values
          # Convert observables to the format expected by compute_expectation_values
        # compute_expectation_values expects List[Union[cirq.PauliSum, cirq.PauliString]]
        converted_observables = []
        for obs in observables:
            if isinstance(obs, cirq.PauliSum):
                converted_observables.append(obs)
            else:
                # If it's already a PauliString or other format, keep as is
                converted_observables.append(obs)
        
        # Use the quantum simulation core expectation function
        results = compute_expectation_values(circuits, converted_observables)
        
        return results


# Convenience functions for easy usage
def create_lie_trotter_simulator() -> TrotterSimulator:
    """Create a simulator using first-order Lie-Trotter."""
    return TrotterSimulator(LieTrotter())


def create_suzuki_trotter_simulator(order: int = 2) -> TrotterSimulator:
    """Create a simulator using Suzuki-Trotter of specified order."""
    return TrotterSimulator(SuzukiTrotter(order))


def compare_methods(hamiltonian: cirq.PauliSum, time: float, 
                   observable: cirq.PauliSum, steps_range: range) -> dict:
    """
    Compare different Trotterization methods.
    
    Args:
        hamiltonian: Hamiltonian to test
        time: Evolution time
        observable: Observable to measure
        steps_range: Range of Trotter steps to test
        
    Returns:
        Dictionary with results for each method
    """
    methods = {
        'Lie-Trotter': create_lie_trotter_simulator(),
        'Suzuki-2': create_suzuki_trotter_simulator(2),
        'Suzuki-4': create_suzuki_trotter_simulator(4)
    }
    
    results = {}
    
    for method_name, simulator in methods.items():
        method_results = []
        for steps in steps_range:
            circuit = simulator.method.evolve(hamiltonian, time, steps)
            exp_val = simulator.compute_expectation_values([circuit], [observable])
            method_results.append(exp_val[0, 0])
        results[method_name] = method_results
    
    return results
