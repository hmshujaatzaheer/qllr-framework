"""
Tests to achieve 100% code coverage.
"""

import pytest
import cmath
from qllr.core.syntax import (
    Term, Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit, Inl, Inr, Case
)
from qllr.core.types import (
    Type, QubitType, UnitType, BoolType, TypeVariable,
    LinearArrow, TensorProduct, SumType, BangType, ParagraphType,
    SharpType, ForallType, types_equal
)
from qllr.typing.typechecker import TypeChecker, TypeCheckError, Context
from qllr.typing.orthogonality import OrthogonalityChecker, Difference, check_orthogonality
from qllr.compilation.circuit import QuantumCircuit, Gate, GateType
from qllr.compilation.circuit_extraction import CircuitExtractor, CircuitExtractionError


class TestSuperpositionExtractionEdgeCases:
    """Test superposition extraction edge cases."""
    
    def test_extract_superposition_non_equal_amplitudes(self):
        """Test extracting superposition with non-equal amplitudes."""
        extractor = CircuitExtractor()
        # Amplitudes not close to 1/√2
        term = Superposition(0.9, Ket0(), 0.1, Ket1())
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
    
    def test_extract_superposition_with_imaginary(self):
        """Test extracting superposition with imaginary amplitudes."""
        extractor = CircuitExtractor()
        sqrt2 = 1/cmath.sqrt(2)
        # Add imaginary component
        term = Superposition(sqrt2, Ket0(), sqrt2 * 1j, Ket1())
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1


class TestQCtrlExtractionEdgeCases:
    """Test qctrl extraction edge cases."""
    
    def test_extract_qctrl_differ_by_z(self):
        """Test qctrl extraction when branches differ by Z."""
        extractor = CircuitExtractor()
        sqrt2 = 1/cmath.sqrt(2)
        # |+⟩ and |-⟩ differ by Z
        plus = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        minus = Superposition(sqrt2, Ket0(), -sqrt2, Ket1())
        term = QCtrl(Ket0(), plus, minus)
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
    
    def test_extract_qctrl_general_case(self):
        """Test qctrl extraction with general branches."""
        extractor = CircuitExtractor()
        # Branches that don't differ by simple X or Z
        term = QCtrl(Ket0(), UnitaryApp(UnitaryGate.H, Ket0()), UnitaryApp(UnitaryGate.X, Ket0()))
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1


class TestTypesBaseClassMethods:
    """Test Type base class abstract method implementations."""
    
    def test_type_is_abstract(self):
        """Test that Type is abstract."""
        assert hasattr(Type, '__abstractmethods__')


class TestSyntaxEdgeCases:
    """Test syntax edge cases for coverage."""
    
    def test_case_is_value(self):
        """Test Case is_value returns False."""
        c = Case(Inl(Ket0()), "x", Variable("x"), "y", Variable("y"))
        assert c.is_value() == False
    
    def test_let_tensor_is_value(self):
        """Test LetTensor is_value returns False."""
        lt = LetTensor("x", "y", TensorPair(Ket0(), Ket1()), Variable("x"))
        assert lt.is_value() == False
    
    def test_let_bang_is_value(self):
        """Test LetBang is_value returns False."""
        lb = LetBang("x", BangIntro(Ket0()), Variable("x"))
        assert lb.is_value() == False
    
    def test_measurement_is_value(self):
        """Test Measurement is_value returns False."""
        m = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        assert m.is_value() == False
    
    def test_unitary_app_is_value(self):
        """Test UnitaryApp is_value returns False."""
        u = UnitaryApp(UnitaryGate.H, Ket0())
        # UnitaryApp is not a value (it's a computation)
        assert u.is_value() == False
    
    def test_qctrl_is_value(self):
        """Test QCtrl is_value returns False."""
        q = QCtrl(Ket0(), Ket0(), Ket1())
        assert q.is_value() == False


class TestTypeCheckerEdgeCases:
    """Test type checker edge cases."""
    
    def test_check_application_non_function(self):
        """Test application where func is not a function type."""
        tc = TypeChecker()
        # |0⟩ |1⟩ - trying to apply a qubit
        term = Application(Ket0(), Ket1())
        with pytest.raises(TypeCheckError, match="function type"):
            tc.check(term)
    
    def test_infer_param_type_returns_none(self):
        """Test parameter type inference when it can't infer."""
        tc = TypeChecker()
        # Inference might return None for complex cases
        result = tc._infer_param_type("x", Unit())
        assert result is None


class TestOrthogonalityEdgeCases:
    """Test orthogonality checker edge cases."""
    
    def test_check_orthogonality_function(self):
        """Test check_orthogonality convenience function."""
        result = check_orthogonality(Ket0(), Ket1())
        assert result == True
    
    def test_structurally_equal_with_unit(self):
        """Test _structurally_equal with Unit terms."""
        checker = OrthogonalityChecker()
        assert checker._structurally_equal(Unit(), Unit()) == True
    
    def test_structurally_equal_with_new(self):
        """Test _structurally_equal with New terms."""
        checker = OrthogonalityChecker()
        assert checker._structurally_equal(New(), New()) == True
    
    def test_find_differences_variable(self):
        """Test _find_differences with different variables."""
        checker = OrthogonalityChecker()
        v1 = Variable("x")
        v2 = Variable("y")
        diffs = checker._find_differences(v1, v2, [])
        assert len(diffs) >= 1


class TestCircuitExtractionVariableHandling:
    """Test circuit extraction variable handling."""
    
    def test_extract_abstraction_applies_correctly(self):
        """Test that abstraction extraction works correctly."""
        extractor = CircuitExtractor()
        # Simple identity-like abstraction
        term = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
    
    def test_extract_let_bang_extracts_body(self):
        """Test let-bang extraction extracts the body."""
        extractor = CircuitExtractor()
        term = LetBang("x", BangIntro(Ket0()), UnitaryApp(UnitaryGate.H, Variable("x")))
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1


class TestContextOperations:
    """Test Context operations for coverage."""
    
    def test_context_is_empty_linear_with_unrestricted(self):
        """Test is_empty_linear when only unrestricted vars exist."""
        ctx = Context().add_unrestricted("x", QubitType())
        assert ctx.is_empty_linear() == True
    
    def test_context_linear_vars(self):
        """Test linear_vars returns correct set."""
        ctx = Context().add_linear("x", QubitType()).add_linear("y", QubitType())
        assert ctx.linear_vars() == {"x", "y"}


class TestTypesEqualAllBranches:
    """Test types_equal covers all branches."""
    
    def test_types_equal_different_constructors(self):
        """Test types_equal with different type constructors."""
        assert not types_equal(QubitType(), LinearArrow(QubitType(), QubitType()))
        assert not types_equal(TensorProduct(QubitType(), QubitType()), SumType(QubitType(), QubitType()))
        assert not types_equal(BangType(QubitType()), ParagraphType(QubitType()))
    
    def test_types_equal_nested_forall(self):
        """Test types_equal with nested forall."""
        f1 = ForallType("X", ForallType("Y", TypeVariable("X")))
        f2 = ForallType("A", ForallType("B", TypeVariable("A")))
        assert types_equal(f1, f2)


class TestDifferenceStr:
    """Test Difference string representation."""
    
    def test_difference_str(self):
        """Test Difference __str__ method."""
        d = Difference(["path"], Ket0(), Ket1())
        s = str(d)
        # Just verify it doesn't crash
        assert s is not None
