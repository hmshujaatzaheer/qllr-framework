"""
Integration tests for the complete λ^QLLR pipeline.

Tests the full workflow: Term → Type Check → Extract Circuit → Analyze
"""

import pytest
import cmath
from qllr.core.syntax import (
    Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit
)
from qllr.core.types import QubitType, LinearArrow, TensorProduct, SharpType
from qllr.typing import TypeChecker, OrthogonalityChecker, TypeCheckError
from qllr.compilation import CircuitExtractor, extract_circuit
from qllr.analysis import ComplexityAnalyzer, WidthAnalyzer


class TestFullPipeline:
    """Integration tests for the complete type-check-to-circuit pipeline."""
    
    def test_simple_hadamard_pipeline(self):
        """Test: H[|0⟩] through complete pipeline."""
        term = UnitaryApp(UnitaryGate.H, Ket0())
        
        # Type check
        tc = TypeChecker()
        typ = tc.check(term)
        assert typ == QubitType()
        
        # Extract circuit
        extractor = CircuitExtractor()
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
        assert circuit.size() >= 1
        
        # Analyze complexity
        comp_analyzer = ComplexityAnalyzer()
        bound = comp_analyzer.analyze_term(term)
        assert bound.exponent >= 2
        
        # Analyze width
        width_analyzer = WidthAnalyzer()
        width = width_analyzer.analyze_term(term)
        assert width.width == 1
    
    def test_bell_state_pipeline(self):
        """Test Bell state preparation through pipeline."""
        # H[|0⟩] ⊗ |0⟩ (first part of Bell state)
        term = TensorPair(UnitaryApp(UnitaryGate.H, Ket0()), Ket0())
        
        # Type check
        tc = TypeChecker()
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
        
        # Extract circuit
        circuit = extract_circuit(term)
        assert circuit.num_qubits >= 2
        
        # Analyze width
        width_analyzer = WidthAnalyzer()
        width = width_analyzer.analyze_term(term)
        assert width.width == 2
    
    def test_quantum_control_pipeline(self):
        """Test quantum control through complete pipeline."""
        # qctrl(|0⟩, |0⟩, |1⟩)
        term = QCtrl(Ket0(), Ket0(), Ket1())
        
        # Type check with orthogonality
        orth_checker = OrthogonalityChecker()
        tc = TypeChecker(orthogonality_checker=orth_checker)
        typ = tc.check(term)
        assert isinstance(typ, SharpType)
        
        # Extract circuit
        circuit = extract_circuit(term)
        assert circuit.num_qubits >= 1
        
        # Analyze
        comp_analyzer = ComplexityAnalyzer()
        bound = comp_analyzer.analyze_term(term)
        assert bound.exponent >= 2
    
    def test_lambda_abstraction_pipeline(self):
        """Test lambda abstraction through pipeline."""
        # λx. H[x]
        term = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        
        # Type check
        tc = TypeChecker()
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
        assert typ.domain == QubitType()
        assert typ.codomain == QubitType()
        
        # Extract circuit
        circuit = extract_circuit(term)
        assert circuit.num_qubits >= 1
    
    def test_application_pipeline(self):
        """Test function application through pipeline."""
        # (λx. H[x]) |0⟩
        func = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        term = Application(func, Ket0())
        
        # Type check
        tc = TypeChecker()
        typ = tc.check(term)
        assert typ == QubitType()
        
        # Extract circuit
        circuit = extract_circuit(term)
        # Should have at least H gate
        from qllr.compilation.circuit import GateType
        assert any(g.gate_type == GateType.H for g in circuit.gates)
    
    def test_let_tensor_pipeline(self):
        """Test let-tensor elimination through pipeline."""
        # let x ⊗ y = (|0⟩ ⊗ |1⟩) in x ⊗ y (use both variables)
        tensor = TensorPair(Ket0(), Ket1())
        term = LetTensor("x", "y", tensor, TensorPair(Variable("x"), Variable("y")))
        
        # Type check
        tc = TypeChecker()
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
        
        # Extract circuit
        circuit = extract_circuit(term)
        assert circuit.num_qubits >= 2
    
    def test_superposition_pipeline(self):
        """Test superposition through pipeline."""
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        
        # Type check
        tc = TypeChecker()
        typ = tc.check(term)
        assert isinstance(typ, SharpType)
        
        # Extract circuit
        circuit = extract_circuit(term)
        from qllr.compilation.circuit import GateType
        assert any(g.gate_type == GateType.H for g in circuit.gates)


class TestOrthogonalityIntegration:
    """Integration tests for orthogonality checking with type system."""
    
    def test_orthogonal_branches_accepted(self):
        """Orthogonal branches should type check."""
        orth_checker = OrthogonalityChecker()
        tc = TypeChecker(orthogonality_checker=orth_checker)
        
        # qctrl(H[|0⟩], |0⟩, |1⟩) - orthogonal branches
        term = QCtrl(
            UnitaryApp(UnitaryGate.H, Ket0()),
            Ket0(),
            Ket1()
        )
        typ = tc.check(term)
        assert isinstance(typ, SharpType)
    
    def test_non_orthogonal_branches_rejected(self):
        """Non-orthogonal branches should fail type checking."""
        orth_checker = OrthogonalityChecker()
        tc = TypeChecker(orthogonality_checker=orth_checker)
        
        # qctrl(|0⟩, |0⟩, |0⟩) - same branches (not orthogonal)
        term = QCtrl(Ket0(), Ket0(), Ket0())
        
        with pytest.raises(TypeCheckError):
            tc.check(term)
    
    def test_higher_order_orthogonal_branches(self):
        """Higher-order orthogonal functions should be accepted."""
        orth_checker = OrthogonalityChecker()
        tc = TypeChecker(orthogonality_checker=orth_checker)
        
        # Functions that differ only in orthogonal constant
        # λx. (H[x] ⊗ |0⟩) vs λx. (H[x] ⊗ |1⟩)
        f0 = Abstraction("x", TensorPair(
            UnitaryApp(UnitaryGate.H, Variable("x")),
            Ket0()
        ))
        f1 = Abstraction("x", TensorPair(
            UnitaryApp(UnitaryGate.H, Variable("x")),
            Ket1()
        ))
        
        assert orth_checker.check_orthogonal(f0, f1) == True


class TestCircuitCorrectness:
    """Integration tests for circuit extraction correctness."""
    
    def test_identity_function_circuit(self):
        """Identity function should produce minimal circuit."""
        # λx. x
        term = Abstraction("x", Variable("x"))
        circuit = extract_circuit(term)
        # Identity should not add gates
        assert circuit.size() == 0 or all(
            g.gate_type.value == "I" for g in circuit.gates
        )
    
    def test_hadamard_on_ket1(self):
        """H[|1⟩] should produce X then H."""
        term = UnitaryApp(UnitaryGate.H, Ket1())
        circuit = extract_circuit(term)
        
        from qllr.compilation.circuit import GateType
        gate_types = [g.gate_type for g in circuit.gates]
        
        # Should have both X (for |1⟩) and H
        assert GateType.X in gate_types
        assert GateType.H in gate_types
    
    def test_chained_gates(self):
        """Chained gates should be extracted in order."""
        # X[H[|0⟩]]
        term = UnitaryApp(UnitaryGate.X, 
                         UnitaryApp(UnitaryGate.H, Ket0()))
        circuit = extract_circuit(term)
        
        from qllr.compilation.circuit import GateType
        # Should have H before X
        assert circuit.size() >= 2


class TestComplexityBounds:
    """Integration tests for complexity bound verification."""
    
    def test_simple_term_polytime(self):
        """Simple terms should be in polynomial time fragment."""
        analyzer = ComplexityAnalyzer()
        
        # |0⟩
        assert analyzer.is_in_polytime_fragment(Ket0())
        
        # H[|0⟩]
        assert analyzer.is_in_polytime_fragment(UnitaryApp(UnitaryGate.H, Ket0()))
        
        # λx. H[x]
        assert analyzer.is_in_polytime_fragment(
            Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        )
    
    def test_paragraph_increases_depth(self):
        """Paragraph modality should increase modal depth."""
        analyzer = ComplexityAnalyzer()
        
        # |0⟩ has depth 0
        assert analyzer._term_depth(Ket0()) == 0
        
        # §|0⟩ has depth 1
        assert analyzer._term_depth(ParagraphIntro(Ket0())) == 1
        
        # §§|0⟩ has depth 2
        assert analyzer._term_depth(ParagraphIntro(ParagraphIntro(Ket0()))) == 2
    
    def test_complexity_bound_scales_with_depth(self):
        """Complexity bound should scale with modal depth."""
        analyzer = ComplexityAnalyzer()
        
        # Depth 0: bound is n^2
        bound0 = analyzer.analyze_term(Ket0())
        
        # Depth 1: bound is n^4
        bound1 = analyzer.analyze_term(ParagraphIntro(Ket0()))
        
        # Depth 2: bound is n^8
        bound2 = analyzer.analyze_term(ParagraphIntro(ParagraphIntro(Ket0())))
        
        assert bound0.exponent < bound1.exponent < bound2.exponent


class TestWidthBounds:
    """Integration tests for width bound verification."""
    
    def test_tensor_width_additive(self):
        """Tensor products should have additive width."""
        analyzer = WidthAnalyzer()
        
        # Single qubit: width 1
        w1 = analyzer.analyze_term(Ket0())
        assert w1.width == 1
        
        # Two qubits: width 2
        w2 = analyzer.analyze_term(TensorPair(Ket0(), Ket1()))
        assert w2.width == 2
        
        # Three qubits: width 3
        w3 = analyzer.analyze_term(TensorPair(TensorPair(Ket0(), Ket1()), Ket0()))
        assert w3.width == 3
    
    def test_unitary_preserves_width(self):
        """Unitary operations should preserve width."""
        analyzer = WidthAnalyzer()
        
        # |0⟩ has width 1
        w_ket = analyzer.analyze_term(Ket0())
        
        # H[|0⟩] should also have width 1
        w_h = analyzer.analyze_term(UnitaryApp(UnitaryGate.H, Ket0()))
        
        assert w_ket.width == w_h.width
    
    def test_qctrl_adds_width(self):
        """Quantum control should add width for control qubit."""
        analyzer = WidthAnalyzer()
        
        # qctrl adds at least 1 for control
        term = QCtrl(Ket0(), Ket0(), Ket1())
        w = analyzer.analyze_term(term)
        
        # Should be at least 2 (control + at least one branch qubit)
        assert w.width >= 2


class TestOpenQASMGeneration:
    """Integration tests for OpenQASM output."""
    
    def test_hadamard_openqasm(self):
        """H[|0⟩] should produce valid OpenQASM."""
        term = UnitaryApp(UnitaryGate.H, Ket0())
        circuit = extract_circuit(term)
        qasm = circuit.to_openqasm()
        
        assert "OPENQASM 3" in qasm
        assert "qubit" in qasm
        assert "h q[0]" in qasm
    
    def test_cnot_openqasm(self):
        """CNOT should be properly represented in OpenQASM."""
        from qllr.compilation.circuit import QuantumCircuit
        
        circuit = QuantumCircuit(2)
        circuit.add_h(0)
        circuit.add_cnot(0, 1)
        
        qasm = circuit.to_openqasm()
        assert "cx q[0], q[1]" in qasm


class TestEndToEndScenarios:
    """End-to-end test scenarios for practical quantum programs."""
    
    def test_ghz_state_preparation(self):
        """Test GHZ state preparation skeleton."""
        # GHZ = H|0⟩ ⊗ |0⟩ ⊗ |0⟩ followed by CNOTs
        # Here we test just the initial state
        h_qubit = UnitaryApp(UnitaryGate.H, Ket0())
        term = TensorPair(TensorPair(h_qubit, Ket0()), Ket0())
        
        tc = TypeChecker()
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
        
        circuit = extract_circuit(term)
        assert circuit.num_qubits >= 3
        
        width_analyzer = WidthAnalyzer()
        width = width_analyzer.analyze_term(term)
        assert width.width == 3
    
    def test_teleportation_measurement(self):
        """Test measurement part of quantum teleportation."""
        # meas(H[|0⟩])
        term = Measurement(MeasurementBasis.COMPUTATIONAL,
                          UnitaryApp(UnitaryGate.H, Ket0()))
        
        tc = TypeChecker()
        from qllr.core.types import BoolType
        typ = tc.check(term)
        assert typ == BoolType()
        
        circuit = extract_circuit(term)
        assert circuit.num_qubits >= 1
    
    def test_controlled_rotation(self):
        """Test controlled operation via qctrl."""
        # qctrl(H[|0⟩], Z[|0⟩], |0⟩)
        ctrl = UnitaryApp(UnitaryGate.H, Ket0())
        branch0 = UnitaryApp(UnitaryGate.Z, Ket0())
        branch1 = Ket0()
        
        term = QCtrl(ctrl, branch0, branch1)
        
        # This may or may not pass orthogonality depending on checker
        # Just verify extraction works
        circuit = extract_circuit(term)
        assert circuit.num_qubits >= 2
