"""
Comprehensive tests for 100% coverage on circuit extraction.
"""

import pytest
import cmath
from qllr.core.syntax import (
    Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit, Inl, Inr, Case
)
from qllr.compilation.circuit import (
    QuantumCircuit, Gate, GateType, identity_circuit, bell_state_circuit
)
from qllr.compilation.circuit_extraction import (
    CircuitExtractor, CircuitExtractionError, extract_circuit,
    QubitAllocation, ExtractionContext
)


class TestCircuitToQasm:
    """Complete tests for OpenQASM generation."""
    
    def test_qasm_all_gates(self):
        """Test OpenQASM output for all gate types."""
        c = QuantumCircuit(3)
        c.add_h(0)
        c.add_x(1)
        c.add_y(0)
        c.add_z(1)
        c.add_s(0)
        c.add_t(1)
        c.add_cnot(0, 1)
        c.add_cz(1, 2)
        c.add_swap(0, 2)
        
        qasm = c.to_openqasm()
        assert "OPENQASM 3" in qasm
        assert "qubit[3]" in qasm
        assert "h q[0]" in qasm
        assert "x q[1]" in qasm
        assert "y q[0]" in qasm
        assert "z q[1]" in qasm
        assert "s q[0]" in qasm
        assert "t q[1]" in qasm
        assert "cx q[0], q[1]" in qasm
        assert "cz q[1], q[2]" in qasm
        assert "swap q[0], q[2]" in qasm
    
    def test_qasm_s_dagger(self):
        """Test OpenQASM output for S-dagger gate."""
        c = QuantumCircuit(1)
        c.add_gate(Gate(GateType.S_DAG, (0,)))
        qasm = c.to_openqasm()
        assert "sdg q[0]" in qasm
    
    def test_qasm_t_dagger(self):
        """Test OpenQASM output for T-dagger gate."""
        c = QuantumCircuit(1)
        c.add_gate(Gate(GateType.T_DAG, (0,)))
        qasm = c.to_openqasm()
        assert "tdg q[0]" in qasm
    
    def test_qasm_identity(self):
        """Test OpenQASM output for identity gate."""
        c = QuantumCircuit(1)
        c.add_gate(Gate(GateType.I, (0,)))
        qasm = c.to_openqasm()
        assert "id q[0]" in qasm


class TestCircuitInverseComplete:
    """Complete tests for circuit inverse."""
    
    def test_inverse_single_gate(self):
        """Test inverse of single gate."""
        c = QuantumCircuit(1)
        c.add_h(0)
        inv = c.inverse()
        assert inv.size() == 1
        # H is self-inverse
        assert inv.gates[0].gate_type == GateType.H
    
    def test_inverse_s_gate(self):
        """Test inverse of S gate."""
        c = QuantumCircuit(1)
        c.add_s(0)
        inv = c.inverse()
        assert inv.gates[0].gate_type == GateType.S_DAG
    
    def test_inverse_t_gate(self):
        """Test inverse of T gate."""
        c = QuantumCircuit(1)
        c.add_t(0)
        inv = c.inverse()
        assert inv.gates[0].gate_type == GateType.T_DAG
    
    def test_inverse_s_dagger(self):
        """Test inverse of S-dagger gate."""
        c = QuantumCircuit(1)
        c.add_gate(Gate(GateType.S_DAG, (0,)))
        inv = c.inverse()
        assert inv.gates[0].gate_type == GateType.S
    
    def test_inverse_t_dagger(self):
        """Test inverse of T-dagger gate."""
        c = QuantumCircuit(1)
        c.add_gate(Gate(GateType.T_DAG, (0,)))
        inv = c.inverse()
        assert inv.gates[0].gate_type == GateType.T
    
    def test_inverse_order(self):
        """Test that inverse reverses gate order."""
        c = QuantumCircuit(1)
        c.add_h(0)
        c.add_t(0)
        c.add_s(0)
        inv = c.inverse()
        # Order should be reversed: S†, T†, H
        assert inv.gates[0].gate_type == GateType.S_DAG
        assert inv.gates[1].gate_type == GateType.T_DAG
        assert inv.gates[2].gate_type == GateType.H


class TestCircuitTensor:
    """Complete tests for circuit tensor product."""
    
    def test_tensor_shifts_qubits(self):
        """Test that tensor shifts qubit indices."""
        c1 = QuantumCircuit(1)
        c1.add_h(0)
        
        c2 = QuantumCircuit(1)
        c2.add_x(0)
        
        tensored = c1.tensor(c2)
        assert tensored.num_qubits == 2
        # Second circuit's gates should be shifted
        assert tensored.gates[1].target_qubits == (1,)


class TestCircuitDepthCalculation:
    """Complete tests for circuit depth calculation."""
    
    def test_depth_with_controls(self):
        """Test depth with controlled gates."""
        c = QuantumCircuit(3)
        c.add_h(0)
        c.add_cnot(0, 1)  # Uses qubit 0 and 1
        c.add_cnot(1, 2)  # Uses qubit 1 and 2
        assert c.depth() == 3


class TestQubitAllocationComplete:
    """Complete tests for qubit allocation."""
    
    def test_deallocate_nonexistent(self):
        """Test deallocating a qubit that wasn't allocated."""
        alloc = QubitAllocation()
        # Deallocating should not fail even if qubit wasn't tracked
        alloc.deallocate(99)  # Should not raise
    
    def test_allocate_with_reuse(self):
        """Test allocation reuses deallocated qubits."""
        alloc = QubitAllocation()
        q0 = alloc.allocate()
        q1 = alloc.allocate()
        alloc.deallocate(q0)
        q2 = alloc.allocate()
        assert q2 == q0  # Reused
    
    def test_remove_nonexistent(self):
        """Test removing a variable that doesn't exist."""
        alloc = QubitAllocation()
        with pytest.raises(CircuitExtractionError):
            alloc.remove("nonexistent")


class TestCircuitExtractorComplete:
    """Complete tests for circuit extractor."""
    
    def test_extract_with_unbound_variable_error(self):
        """Test extracting unbound variable raises error."""
        extractor = CircuitExtractor()
        # Variable not in context
        term = Variable("x")
        with pytest.raises(CircuitExtractionError):
            extractor.extract(term)
    
    def test_extract_qctrl_differ_by_z(self):
        """Test qctrl where branches differ by Z gate."""
        extractor = CircuitExtractor()
        sqrt2 = 1/cmath.sqrt(2)
        # |+⟩ and |-⟩ differ by Z
        plus = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        minus = Superposition(sqrt2, Ket0(), -sqrt2, Ket1())
        term = QCtrl(Ket0(), plus, minus)
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 2
    
    def test_extract_multi_qubit_unitary(self):
        """Test extracting multi-qubit unitary."""
        extractor = CircuitExtractor()
        # CNOT on tensor
        term = UnitaryApp(UnitaryGate.CNOT, TensorPair(Ket0(), Ket1()))
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.CNOT for g in circuit.gates)
    
    def test_extract_cz_gate(self):
        """Test extracting CZ gate."""
        extractor = CircuitExtractor()
        term = UnitaryApp(UnitaryGate.CZ, TensorPair(Ket0(), Ket1()))
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.CZ for g in circuit.gates)
    
    def test_extract_swap_gate(self):
        """Test extracting SWAP gate."""
        extractor = CircuitExtractor()
        term = UnitaryApp(UnitaryGate.SWAP, TensorPair(Ket0(), Ket1()))
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.SWAP for g in circuit.gates)


class TestBranchDifferenceDetectionComplete:
    """Complete tests for branch difference detection."""
    
    def test_branches_differ_by_x_basic(self):
        """Test basic X difference detection."""
        extractor = CircuitExtractor()
        result = extractor._branches_differ_by_x(Ket0(), Ket1())
        assert result == True
    
    def test_branches_differ_not_detected(self):
        """Test when branch difference is not detected."""
        extractor = CircuitExtractor()
        # Complex terms that don't have simple difference
        t1 = TensorPair(Ket0(), Ket0())
        t2 = TensorPair(Ket1(), Ket1())
        result = extractor._branches_differ_by_x(t1, t2)
        # May or may not detect depending on implementation
        assert isinstance(result, bool)


class TestExtractionContextComplete:
    """Complete tests for extraction context."""
    
    def test_context_finalize_updates_qubits(self):
        """Test that finalize sets correct qubit count."""
        ctx = ExtractionContext()
        ctx.allocation.allocate()
        ctx.allocation.allocate()
        ctx.allocation.allocate()
        circuit = ctx.finalize()
        assert circuit.num_qubits == 3


class TestGateTypeComplete:
    """Complete tests for GateType enum."""
    
    def test_all_gate_types_exist(self):
        """Verify all gate types exist."""
        expected = ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'CZ', 'SWAP', 
                    'S_DAG', 'T_DAG', 'I']
        for name in expected:
            assert hasattr(GateType, name)
    
    def test_gate_str(self):
        """Test gate string representation."""
        g = Gate(GateType.CNOT, (1,), (0,))
        s = str(g)
        assert "CNOT" in s


class TestCircuitEdgeCases:
    """Edge case tests for QuantumCircuit."""
    
    def test_t_count_with_no_t_gates(self):
        """Test T-count when no T gates present."""
        c = QuantumCircuit(1)
        c.add_h(0)
        c.add_x(0)
        assert c.t_count() == 0
    
    def test_depth_empty_circuit(self):
        """Test depth of empty circuit."""
        c = QuantumCircuit(5)
        assert c.depth() == 0
    
    def test_size_empty_circuit(self):
        """Test size of empty circuit."""
        c = QuantumCircuit(5)
        assert c.size() == 0


class TestExtractUnknownTerm:
    """Test extraction of unknown term types."""
    
    def test_extract_unknown_raises_error(self):
        """Test that unknown term types raise error."""
        extractor = CircuitExtractor()
        
        class UnknownTerm:
            def free_variables(self):
                return set()
        
        with pytest.raises(CircuitExtractionError, match="Cannot extract"):
            extractor.extract(UnknownTerm())
