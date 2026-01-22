"""
Unit tests for λ^QLLR circuit compilation.
"""

import pytest
from qllr.core.syntax import (
    Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit
)
from qllr.compilation.circuit import (
    QuantumCircuit, Gate, GateType, identity_circuit, bell_state_circuit
)
from qllr.compilation.circuit_extraction import (
    CircuitExtractor, CircuitExtractionError, extract_circuit,
    QubitAllocation, ExtractionContext
)
import cmath


class TestGate:
    """Tests for quantum gates."""
    
    def test_single_qubit_gate(self):
        g = Gate(GateType.H, (0,))
        assert g.arity() == 1
        assert g.all_qubits() == {0}
        assert g.is_controlled() == False
    
    def test_two_qubit_gate(self):
        g = Gate(GateType.CNOT, (1,), (0,))
        assert g.arity() == 2
        assert g.all_qubits() == {0, 1}
        assert g.is_controlled() == True
    
    def test_gate_str(self):
        g = Gate(GateType.H, (0,))
        assert "H" in str(g)
        
        cnot = Gate(GateType.CNOT, (1,), (0,))
        assert "CNOT" in str(cnot)
    
    def test_gate_with_parameters(self):
        g = Gate(GateType.H, (0,), (), (0.5,))
        assert g.parameters == (0.5,)


class TestQuantumCircuit:
    """Tests for quantum circuits."""
    
    def test_empty_circuit(self):
        c = QuantumCircuit(2)
        assert c.num_qubits == 2
        assert c.size() == 0
        assert c.depth() == 0
    
    def test_add_gate(self):
        c = QuantumCircuit(2)
        c.add_gate(Gate(GateType.H, (0,)))
        assert c.size() == 1
    
    def test_add_h(self):
        c = QuantumCircuit(1)
        c.add_h(0)
        assert c.size() == 1
        assert c.gates[0].gate_type == GateType.H
    
    def test_add_x(self):
        c = QuantumCircuit(1)
        c.add_x(0)
        assert c.gates[0].gate_type == GateType.X
    
    def test_add_y(self):
        c = QuantumCircuit(1)
        c.add_y(0)
        assert c.gates[0].gate_type == GateType.Y
    
    def test_add_z(self):
        c = QuantumCircuit(1)
        c.add_z(0)
        assert c.gates[0].gate_type == GateType.Z
    
    def test_add_s(self):
        c = QuantumCircuit(1)
        c.add_s(0)
        assert c.gates[0].gate_type == GateType.S
    
    def test_add_t(self):
        c = QuantumCircuit(1)
        c.add_t(0)
        assert c.gates[0].gate_type == GateType.T
    
    def test_add_cnot(self):
        c = QuantumCircuit(2)
        c.add_cnot(0, 1)
        assert c.gates[0].gate_type == GateType.CNOT
        assert c.gates[0].control_qubits == (0,)
        assert c.gates[0].target_qubits == (1,)
    
    def test_add_cz(self):
        c = QuantumCircuit(2)
        c.add_cz(0, 1)
        assert c.gates[0].gate_type == GateType.CZ
    
    def test_add_swap(self):
        c = QuantumCircuit(2)
        c.add_swap(0, 1)
        assert c.gates[0].gate_type == GateType.SWAP
    
    def test_add_controlled(self):
        c = QuantumCircuit(2)
        c.add_controlled(GateType.Z, 0, 1)
        assert c.gates[0].control_qubits == (0,)
    
    def test_depth_single_qubit(self):
        c = QuantumCircuit(1)
        c.add_h(0)
        c.add_x(0)
        c.add_z(0)
        assert c.depth() == 3
    
    def test_depth_parallel(self):
        c = QuantumCircuit(2)
        c.add_h(0)
        c.add_h(1)
        assert c.depth() == 1  # Can be parallel
    
    def test_depth_sequential(self):
        c = QuantumCircuit(2)
        c.add_h(0)
        c.add_cnot(0, 1)
        c.add_h(1)
        assert c.depth() == 3
    
    def test_width(self):
        c = QuantumCircuit(5)
        assert c.width() == 5
    
    def test_t_count(self):
        c = QuantumCircuit(1)
        c.add_t(0)
        c.add_h(0)
        c.add_t(0)
        assert c.t_count() == 2
    
    def test_compose(self):
        c1 = QuantumCircuit(2)
        c1.add_h(0)
        
        c2 = QuantumCircuit(2)
        c2.add_cnot(0, 1)
        
        composed = c1.compose(c2)
        assert composed.size() == 2
    
    def test_compose_different_sizes_error(self):
        c1 = QuantumCircuit(2)
        c2 = QuantumCircuit(3)
        with pytest.raises(ValueError):
            c1.compose(c2)
    
    def test_tensor(self):
        c1 = QuantumCircuit(1)
        c1.add_h(0)
        
        c2 = QuantumCircuit(1)
        c2.add_x(0)
        
        tensored = c1.tensor(c2)
        assert tensored.num_qubits == 2
        assert tensored.size() == 2
    
    def test_inverse(self):
        c = QuantumCircuit(1)
        c.add_h(0)
        c.add_s(0)
        
        inv = c.inverse()
        assert inv.size() == 2
        # S† should be first in inverse
        assert inv.gates[0].gate_type == GateType.S_DAG
    
    def test_openqasm(self):
        c = QuantumCircuit(2)
        c.add_h(0)
        c.add_cnot(0, 1)
        
        qasm = c.to_openqasm()
        assert "OPENQASM 3" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0], q[1]" in qasm
    
    def test_str(self):
        c = QuantumCircuit(2)
        c.add_h(0)
        s = str(c)
        assert "Circuit" in s
        assert "2 qubits" in s
    
    def test_invalid_qubit_index(self):
        c = QuantumCircuit(2)
        with pytest.raises(ValueError):
            c.add_gate(Gate(GateType.H, (5,)))


class TestIdentityCircuit:
    """Tests for identity circuit helper."""
    
    def test_identity_circuit(self):
        c = identity_circuit(3)
        assert c.num_qubits == 3
        assert c.size() == 0
        assert c.input_qubits == (0, 1, 2)
        assert c.output_qubits == (0, 1, 2)


class TestBellStateCircuit:
    """Tests for Bell state circuit helper."""
    
    def test_bell_state_circuit(self):
        c = bell_state_circuit()
        assert c.num_qubits == 2
        assert c.size() == 2
        # First gate is H
        assert c.gates[0].gate_type == GateType.H
        # Second gate is CNOT
        assert c.gates[1].gate_type == GateType.CNOT


class TestQubitAllocation:
    """Tests for qubit allocation during extraction."""
    
    def test_allocate_fresh(self):
        alloc = QubitAllocation()
        q = alloc.allocate()
        assert q == 0
        assert alloc.total_qubits == 1
    
    def test_allocate_multiple(self):
        alloc = QubitAllocation()
        q0 = alloc.allocate()
        q1 = alloc.allocate()
        assert q0 == 0
        assert q1 == 1
        assert alloc.total_qubits == 2
    
    def test_allocate_with_variable(self):
        alloc = QubitAllocation()
        q = alloc.allocate("x")
        assert q == 0
        assert alloc.lookup("x") == 0
    
    def test_deallocate_and_reuse(self):
        alloc = QubitAllocation()
        q0 = alloc.allocate()
        alloc.deallocate(q0)
        q1 = alloc.allocate()
        assert q1 == q0  # Reused
    
    def test_lookup_not_found(self):
        alloc = QubitAllocation()
        with pytest.raises(CircuitExtractionError):
            alloc.lookup("nonexistent")
    
    def test_remove_variable(self):
        alloc = QubitAllocation()
        alloc.allocate("x")
        q = alloc.remove("x")
        assert q == 0
        with pytest.raises(CircuitExtractionError):
            alloc.lookup("x")


class TestExtractionContext:
    """Tests for extraction context."""
    
    def test_finalize(self):
        ctx = ExtractionContext()
        ctx.allocation.allocate()
        ctx.allocation.allocate()
        circuit = ctx.finalize()
        assert circuit.num_qubits == 2


class TestCircuitExtractor:
    """Tests for circuit extraction from terms."""
    
    def test_extract_ket0(self):
        extractor = CircuitExtractor()
        circuit = extractor.extract(Ket0())
        assert circuit.num_qubits == 1
        assert circuit.size() == 0  # |0⟩ is default
    
    def test_extract_ket1(self):
        extractor = CircuitExtractor()
        circuit = extractor.extract(Ket1())
        assert circuit.num_qubits == 1
        assert circuit.size() == 1  # X gate to flip to |1⟩
        assert circuit.gates[0].gate_type == GateType.X
    
    def test_extract_hadamard(self):
        extractor = CircuitExtractor()
        term = UnitaryApp(UnitaryGate.H, Ket0())
        circuit = extractor.extract(term)
        assert circuit.size() == 1
        assert circuit.gates[0].gate_type == GateType.H
    
    def test_extract_pauli_x(self):
        extractor = CircuitExtractor()
        term = UnitaryApp(UnitaryGate.X, Ket0())
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.X for g in circuit.gates)
    
    def test_extract_pauli_y(self):
        extractor = CircuitExtractor()
        term = UnitaryApp(UnitaryGate.Y, Ket0())
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.Y for g in circuit.gates)
    
    def test_extract_pauli_z(self):
        extractor = CircuitExtractor()
        term = UnitaryApp(UnitaryGate.Z, Ket0())
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.Z for g in circuit.gates)
    
    def test_extract_s_gate(self):
        extractor = CircuitExtractor()
        term = UnitaryApp(UnitaryGate.S, Ket0())
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.S for g in circuit.gates)
    
    def test_extract_t_gate(self):
        extractor = CircuitExtractor()
        term = UnitaryApp(UnitaryGate.T, Ket0())
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.T for g in circuit.gates)
    
    def test_extract_qctrl_ket(self):
        extractor = CircuitExtractor()
        # qctrl(|0⟩, |0⟩, |1⟩)
        term = QCtrl(Ket0(), Ket0(), Ket1())
        circuit = extractor.extract(term)
        # Should have CNOT for the controlled-X operation
        assert any(g.gate_type == GateType.CNOT for g in circuit.gates)
    
    def test_extract_tensor_pair(self):
        extractor = CircuitExtractor()
        term = TensorPair(Ket0(), Ket1())
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 2
    
    def test_extract_application_beta(self):
        extractor = CircuitExtractor()
        # (λx. H[x]) |0⟩
        func = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        term = Application(func, Ket0())
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.H for g in circuit.gates)
    
    def test_extract_new(self):
        extractor = CircuitExtractor()
        circuit = extractor.extract(New())
        assert circuit.num_qubits == 1
    
    def test_extract_superposition(self):
        extractor = CircuitExtractor()
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        circuit = extractor.extract(term)
        # Should use H gate for equal superposition
        assert any(g.gate_type == GateType.H for g in circuit.gates)
    
    def test_extract_abstraction(self):
        extractor = CircuitExtractor()
        term = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
    
    def test_extract_let_tensor(self):
        extractor = CircuitExtractor()
        tensor = TensorPair(Ket0(), Ket1())
        term = LetTensor("x", "y", tensor, Variable("x"))
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 2
    
    def test_extract_let_bang(self):
        extractor = CircuitExtractor()
        term = LetBang("x", BangIntro(Ket0()), 
                       UnitaryApp(UnitaryGate.H, Variable("x")))
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.H for g in circuit.gates)
    
    def test_extract_bang_intro(self):
        extractor = CircuitExtractor()
        term = BangIntro(Ket0())
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
    
    def test_extract_paragraph_intro(self):
        extractor = CircuitExtractor()
        term = ParagraphIntro(Ket0())
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
    
    def test_extract_measurement(self):
        extractor = CircuitExtractor()
        term = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
    
    def test_extract_unit(self):
        extractor = CircuitExtractor()
        circuit = extractor.extract(Unit())
        assert circuit.output_qubits == ()
    
    def test_extract_circuit_convenience(self):
        circuit = extract_circuit(Ket0())
        assert circuit.num_qubits == 1


class TestCircuitExtractionComplexTerms:
    """Tests for extracting circuits from complex terms."""
    
    def test_extract_chained_unitaries(self):
        extractor = CircuitExtractor()
        # H[X[|0⟩]]
        term = UnitaryApp(UnitaryGate.H, 
                         UnitaryApp(UnitaryGate.X, Ket0()))
        circuit = extractor.extract(term)
        assert circuit.size() >= 2
    
    def test_extract_nested_tensor(self):
        extractor = CircuitExtractor()
        # (|0⟩ ⊗ |1⟩) ⊗ |0⟩
        inner = TensorPair(Ket0(), Ket1())
        term = TensorPair(inner, Ket0())
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 3
    
    def test_extract_qctrl_with_unitaries(self):
        extractor = CircuitExtractor()
        # qctrl(H[|0⟩], |0⟩, |1⟩)
        ctrl = UnitaryApp(UnitaryGate.H, Ket0())
        term = QCtrl(ctrl, Ket0(), Ket1())
        circuit = extractor.extract(term)
        assert any(g.gate_type == GateType.H for g in circuit.gates)


class TestBranchDifferenceDetection:
    """Tests for branch difference detection in qctrl extraction."""
    
    def test_branches_differ_by_x(self):
        extractor = CircuitExtractor()
        assert extractor._branches_differ_by_x(Ket0(), Ket1()) == True
        assert extractor._branches_differ_by_x(Ket1(), Ket0()) == True
        assert extractor._branches_differ_by_x(Ket0(), Ket0()) == False
    
    def test_branches_differ_by_z(self):
        extractor = CircuitExtractor()
        sqrt2 = 1/cmath.sqrt(2)
        plus = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        minus = Superposition(sqrt2, Ket0(), -sqrt2, Ket1())
        assert extractor._branches_differ_by_z(plus, minus) == True
    
    def test_branches_differ_by_phase(self):
        extractor = CircuitExtractor()
        # Default implementation returns False
        assert extractor._branches_differ_by_phase(Ket0(), Ket1()) == False
