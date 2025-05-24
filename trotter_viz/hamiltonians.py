"""
Quantum Hamiltonian Models for Trotterization

This module implements various quantum spin chain Hamiltonians commonly used
in quantum simulation studies.
"""

import cirq
import numpy as np
from typing import List


class QuantumHamiltonians:
    """Collection of quantum Hamiltonian models for spin chains."""
    
    @staticmethod
    def transverse_field_ising(n_qubits: int, coupling_strengths: List[float]) -> cirq.PauliSum:
        """
        Create a Transverse Field Ising (TFI) model Hamiltonian.
        
        H = -∑ᵢ Z_i Z_{i+1} - ∑ᵢ g_i X_i
        
        Args:
            n_qubits: Number of qubits in the chain
            coupling_strengths: Transverse field strengths for each qubit
            
        Returns:
            Cirq PauliSum representing the TFI Hamiltonian
        """
        qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
        hamiltonian = cirq.PauliSum()
        
        # ZZ interactions
        for i in range(n_qubits - 1):
            hamiltonian -= cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1])
        
        # Transverse field terms
        for i in range(n_qubits):
            hamiltonian -= coupling_strengths[i] * cirq.X(qubits[i])
            
        return hamiltonian
    
    @staticmethod
    def xxz_heisenberg(n_qubits: int, delta: float = 1.0) -> cirq.PauliSum:
        """
        Create an XXZ Heisenberg model Hamiltonian.
        
        H = ∑ᵢ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1})
        
        Args:
            n_qubits: Number of qubits in the chain
            delta: Anisotropy parameter (default 1.0 for isotropic case)
            
        Returns:
            Cirq PauliSum representing the XXZ Hamiltonian
        """
        qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
        hamiltonian = cirq.PauliSum()
        
        for i in range(n_qubits - 1):
            hamiltonian += cirq.X(qubits[i]) * cirq.X(qubits[i + 1])
            hamiltonian += cirq.Y(qubits[i]) * cirq.Y(qubits[i + 1])
            hamiltonian += delta * cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1])
            
        return hamiltonian
    
    @staticmethod
    def xy_model(n_qubits: int, coupling_strengths: List[float], 
                 magnetic_fields: List[float]) -> cirq.PauliSum:
        """
        Create an XY model Hamiltonian.
        
        H = -∑ᵢ J_i (X_i X_{i+1} + Y_i Y_{i+1}) - ∑ᵢ h_i Z_i
        
        Args:
            n_qubits: Number of qubits in the chain
            coupling_strengths: XY coupling strengths between neighboring qubits
            magnetic_fields: Local magnetic field strengths
            
        Returns:
            Cirq PauliSum representing the XY Hamiltonian
        """
        qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
        hamiltonian = cirq.PauliSum()
        
        # XY interactions
        for i in range(n_qubits - 1):
            hamiltonian -= coupling_strengths[i] * cirq.X(qubits[i]) * cirq.X(qubits[i + 1])
            hamiltonian -= coupling_strengths[i] * cirq.Y(qubits[i]) * cirq.Y(qubits[i + 1])
        
        # Magnetic field terms
        for i in range(n_qubits):
            hamiltonian -= magnetic_fields[i] * cirq.Z(qubits[i])
            
        return hamiltonian


def create_observables(n_qubits: int):
    """
    Create common observables for quantum spin chains.
    
    Args:
        n_qubits: Number of qubits in the system
        
    Returns:
        Dictionary of observable operators
    """
    qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    
    observables = {
        'magnetization': sum([cirq.Z(q) for q in qubits]),
        'energy': None,  # Will be set by the specific Hamiltonian
    }
    
    # Add correlation functions
    if n_qubits <= 4:
        # Full correlation for small systems
        observables['correlation'] = 1
        for q in qubits:
            observables['correlation'] *= cirq.Z(q)
    else:
        # Nearest-neighbor correlation for larger systems
        observables['correlation'] = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
    
    # Local observables
    for i, q in enumerate(qubits):
        observables[f'z_{i}'] = cirq.Z(q)
        observables[f'x_{i}'] = cirq.X(q)
        observables[f'y_{i}'] = cirq.Y(q)
    
    return observables


# Additional Hamiltonian functions to match notebook implementation
def TFI_chain(n, gs):
    """
    Transverse Field Ising chain from notebook implementation.
    
    Args:
        n: Number of qubits
        gs: List of transverse field strengths
    
    Returns:
        Cirq PauliSum representing the TFI Hamiltonian
    """
    qs = [cirq.GridQubit(0, i) for i in range(n)]
    ham = cirq.PauliSum()
    for i in range(n - 1):
        ham -= cirq.Z(qs[i]) * cirq.Z(qs[i + 1])
    for i in range(n):
        ham -= gs[i] * cirq.X(qs[i])
    return ham


def XY_hamiltonian(n, gs, ls):
    """
    XY Hamiltonian from notebook implementation.
    
    Args:
        n: Number of qubits
        gs: Coupling strengths
        ls: Local field strengths
    
    Returns:
        Cirq PauliSum representing the XY Hamiltonian
    """
    qs = [cirq.GridQubit(0, i) for i in range(n)]
    ham = cirq.PauliSum()
    for i in range(n - 1):
        ham -= gs[i] * cirq.X(qs[i]) * cirq.X(qs[i + 1])
        ham -= gs[i] * cirq.Y(qs[i]) * cirq.Y(qs[i + 1])
    for i in range(n):
        ham -= ls[i] * cirq.Z(qs[i])
    return ham


def XXZ_chain(n):
    """
    XXZ chain from notebook implementation.
    
    Args:
        n: Number of qubits
    
    Returns:
        Cirq PauliSum representing the XXZ Hamiltonian
    """
    qs = [cirq.GridQubit(0, i) for i in range(n)]
    ham = cirq.PauliSum()
    for i in range(n - 1):
        ham += cirq.X(qs[i]) * cirq.X(qs[i + 1])
        ham += cirq.Y(qs[i]) * cirq.Y(qs[i + 1])
        ham += cirq.Z(qs[i]) * cirq.Z(qs[i + 1])
    return ham
