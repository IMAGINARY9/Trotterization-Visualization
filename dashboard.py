"""
Interactive Dashboard using Streamlit

This module creates a web-based interactive dashboard for exploring
Trotterization methods in quantum Hamiltonian simulation.

Run with: streamlit run dashboard.py
"""

import sys
import os
# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import our modules
from trotter_viz.visualization import TrotterVisualizer, create_dashboard_data
from trotter_viz.hamiltonians import QuantumHamiltonians, create_observables
from trotter_viz.trotterization import (
    create_lie_trotter_simulator,
    create_suzuki_trotter_simulator,
    compare_methods
)


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Trotterization Explorer",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚛️ Interactive Trotterization Explorer")
    st.markdown("""
    Explore quantum Hamiltonian simulation using different Trotterization methods.
    Adjust parameters in the sidebar and see real-time visualizations!
    """)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Simulation Parameters")
    
    # System parameters
    st.sidebar.subheader("Quantum System")
    n_qubits = st.sidebar.slider("Number of Qubits", 2, 6, 4)
    hamiltonian_type = st.sidebar.selectbox(
        "Hamiltonian Model",
        ["TFI", "XXZ", "XY"],
        index=0,
        help="TFI: Transverse Field Ising, XXZ: Heisenberg, XY: XY Model"
    )
    
    # Physical parameters
    st.sidebar.subheader("Physical Parameters")
    coupling_strength = st.sidebar.slider("Coupling Strength", 0.1, 2.0, 1.0, 0.1)
    magnetic_field = st.sidebar.slider("Magnetic Field", 0.0, 2.0, 1.0, 0.1)
    max_time = st.sidebar.slider("Maximum Time", 1.0, 5.0, 3.0, 0.5)
    
    # Simulation parameters
    st.sidebar.subheader("Trotterization Settings")
    max_steps = st.sidebar.slider("Maximum Trotter Steps", 5, 20, 10)
    show_error_analysis = st.sidebar.checkbox("Show Error Analysis", True)
    show_circuit_info = st.sidebar.checkbox("Show Circuit Information", True)
    
    # Create visualizer
    visualizer = TrotterVisualizer()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Time Evolution Comparison")
        
        # Generate main plot
        with st.spinner("Generating time evolution plots..."):
            main_fig = visualizer.plot_time_evolution(
                n_qubits=n_qubits,
                hamiltonian_type=hamiltonian_type,
                max_time=max_time,
                coupling=coupling_strength,
                magnetic_field=magnetic_field,
                max_steps=max_steps
            )
            st.plotly_chart(main_fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 System Information")
        
        # System info
        st.metric("Hilbert Space Dimension", f"2^{n_qubits} = {2**n_qubits}")
        st.metric("Time Points", 40)
        st.metric("Max Trotter Steps", max_steps)
        
        # Method comparison table
        st.subheader("Method Comparison")
        comparison_df = pd.DataFrame({
            'Method': ['Lie-Trotter', 'Suzuki-2', 'Suzuki-4'],
            'Order': [1, 2, 4],
            'Error Scaling': ['O(1/r)', 'O(1/r²)', 'O(1/r⁴)'],
            'Complexity': ['Low', 'Medium', 'High']
        })
        st.dataframe(comparison_df, hide_index=True)
    
    # Error analysis section
    if show_error_analysis:
        st.subheader("🔍 Error Scaling Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            error_fig = visualizer.plot_error_scaling(
                n_qubits=min(n_qubits, 4),  # Limit for computational efficiency
                hamiltonian_type=hamiltonian_type,
                time=2.0,
                max_steps=15
            )
            st.plotly_chart(error_fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            **Key Insights:**
            - **Lie-Trotter**: Simple first-order method with O(1/r) error scaling
            - **Suzuki-2**: Second-order method with O(1/r²) improvement
            - **Suzuki-4**: Fourth-order method for high-precision applications
            
            **Trade-offs:**
            - Higher-order methods are more accurate but computationally expensive
            - Choice depends on required precision vs. available resources
            """)
      # Circuit information
    if show_circuit_info:
        st.subheader("🔄 Quantum Circuit Analysis")
        
        # Create sample circuits for analysis
        if hamiltonian_type == 'TFI':
            sample_ham = QuantumHamiltonians.transverse_field_ising(
                min(n_qubits, 3), [magnetic_field] * min(n_qubits, 3)
            )
        elif hamiltonian_type == 'XXZ':
            sample_ham = QuantumHamiltonians.xxz_heisenberg(min(n_qubits, 3))
        else:
            sample_ham = QuantumHamiltonians.xy_model(
                min(n_qubits, 3), [coupling_strength] * (min(n_qubits, 3)-1), 
                [magnetic_field] * min(n_qubits, 3)
            )
        
        simulators = {
            'Lie-Trotter': create_lie_trotter_simulator(),
            'Suzuki-2': create_suzuki_trotter_simulator(2),
            'Suzuki-4': create_suzuki_trotter_simulator(4)
        }
        
        circuit_data = []
        for method_name, simulator in simulators.items():
            for steps in [1, 3, 5]:
                circuit = simulator.method.evolve(sample_ham, 1.0, steps)
                circuit_data.append({
                    'Method': method_name,
                    'Trotter Steps': steps,
                    'Circuit Depth': len(circuit),
                    'Total Moments': len(circuit),
                    'Estimated Gates': sum(len(moment) for moment in circuit)
                })
        
        circuit_df = pd.DataFrame(circuit_data)
        st.dataframe(circuit_df, hide_index=True)
    
    # Parameter study section
    st.subheader("🧪 Parameter Study")
    
    param_study_col1, param_study_col2 = st.columns(2)
    
    with param_study_col1:
        st.markdown("**Coupling Strength Dependence**")
        coupling_values = np.linspace(0.2, 2.0, 10)
        coupling_fig = visualizer.create_parameter_study(
            'coupling', coupling_values.tolist(), n_qubits=min(n_qubits, 4)
        )
        st.plotly_chart(coupling_fig, use_container_width=True)
    
    with param_study_col2:
        st.markdown("**Evolution Time Dependence**")
        time_values = np.linspace(0.5, 4.0, 10)
        time_fig = visualizer.create_parameter_study(
            'time', time_values.tolist(), n_qubits=min(n_qubits, 4)
        )
        st.plotly_chart(time_fig, use_container_width=True)
    
    # Educational content
    with st.expander("📚 Learn More About Trotterization"):
        st.markdown("""
        ### What is Trotterization?
        
        Trotterization is a fundamental technique in quantum computing for simulating the time evolution 
        of quantum systems. It's based on the mathematical **Lie-Trotter formula**:
        
        $$e^{-i(H_1 + H_2 + \\ldots)t} \\approx \\left(e^{-iH_1 t/r} e^{-iH_2 t/r} \\cdots\\right)^r$$
        
        ### Key Concepts:
        
        1. **Hamiltonian Decomposition**: Complex Hamiltonians are split into simpler, 
           non-commuting terms that can be individually simulated.
        
        2. **Error Scaling**: Different methods have different error behaviors:
           - First-order (Lie-Trotter): Error ∝ 1/r
           - Second-order (Suzuki-2): Error ∝ 1/r²
           - Higher-order methods: Even better scaling
        
        3. **Trade-offs**: More accurate methods require more quantum gates, 
           creating a trade-off between precision and computational cost.
        
        ### Applications:
        - **Quantum Chemistry**: Simulating molecular dynamics
        - **Condensed Matter**: Studying phase transitions
        - **Quantum Machine Learning**: Implementing quantum algorithms
        - **Optimization**: Quantum approximate optimization algorithms (QAOA)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Trotterization Explorer** - Interactive quantum simulation visualization  
    Built with Streamlit and Cirq
    """)


if __name__ == "__main__":
    main()
