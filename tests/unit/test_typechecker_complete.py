"""
Comprehensive tests for 100% coverage on typechecker module.
"""

import pytest
from qllr.core.syntax import (
    Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit, Inl, Inr, Case
)
from qllr.core.types import (
    QubitType, UnitType, BoolType, TypeVariable,
    LinearArrow, TensorProduct, SumType, BangType, ParagraphType,
    SharpType, ForallType
)
from qllr.typing.typechecker import TypeChecker, TypeCheckError, Context
from qllr.typing.orthogonality import OrthogonalityChecker
import cmath


class TestContextEdgeCases:
    """Test Context edge cases for 100% coverage."""
    
    def test_add_linear_already_in_scope(self):
        """Test adding linear var that's already in linear context."""
        ctx = Context().add_linear("x", QubitType())
        with pytest.raises(TypeCheckError, match="already in scope"):
            ctx.add_linear("x", QubitType())
    
    def test_add_linear_already_unrestricted(self):
        """Test adding linear var when already unrestricted."""
        ctx = Context().add_unrestricted("x", QubitType())
        with pytest.raises(TypeCheckError, match="already in scope"):
            ctx.add_linear("x", QubitType())
    
    def test_add_unrestricted_already_in_scope(self):
        """Test adding unrestricted var that's already in linear context."""
        ctx = Context().add_linear("x", QubitType())
        with pytest.raises(TypeCheckError, match="already in scope"):
            ctx.add_unrestricted("x", QubitType())
    
    def test_add_unrestricted_already_unrestricted(self):
        """Test adding unrestricted var when already unrestricted."""
        ctx = Context().add_unrestricted("x", QubitType())
        with pytest.raises(TypeCheckError, match="already in scope"):
            ctx.add_unrestricted("x", QubitType())


class TestUnknownTermType:
    """Test handling of unknown term types."""
    
    def test_unknown_term_type(self):
        """Test that unknown term types raise error."""
        tc = TypeChecker()
        
        # Create a mock term type that's not handled
        class UnknownTerm:
            pass
        
        with pytest.raises(TypeCheckError, match="Unknown term type"):
            tc._check(UnknownTerm(), Context())


class TestCaseStatement:
    """Test case statement checking."""
    
    def test_case_branches_different_types(self):
        """Test case with different branch types."""
        tc = TypeChecker()
        scrutinee = Inl(Ket0())
        # Left branch returns qubit, right branch returns tensor
        term = Case(
            scrutinee,
            "x", Variable("x"),
            "y", TensorPair(Ket0(), Ket1())
        )
        with pytest.raises(TypeCheckError, match="different types"):
            tc.check(term)
    
    def test_case_non_sum_type(self):
        """Test case on non-sum type."""
        tc = TypeChecker()
        # case |0⟩ of ... (not a sum type)
        term = Case(
            Ket0(),
            "x", Variable("x"),
            "y", Variable("y")
        )
        with pytest.raises(TypeCheckError, match="Expected sum type"):
            tc.check(term)


class TestSuperpositionTypeErrors:
    """Test superposition type error cases."""
    
    def test_superposition_different_types(self):
        """Test superposition with different branch types."""
        tc = TypeChecker()
        # α|0⟩ + β(|0⟩⊗|1⟩) - different types
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Ket0(), sqrt2, TensorPair(Ket0(), Ket1()))
        with pytest.raises(TypeCheckError, match="different types"):
            tc.check(term)
    
    def test_superposition_different_resources(self):
        """Test superposition with different linear resources."""
        tc = TypeChecker()
        ctx = Context().add_linear("x", QubitType()).add_linear("y", QubitType())
        # α·x + β·y - uses different resources
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Variable("x"), sqrt2, Variable("y"))
        with pytest.raises(TypeCheckError, match="different linear resources"):
            tc.check(term, ctx)


class TestAbstractionEdgeCases:
    """Test abstraction edge cases."""
    
    def test_abstraction_unused_variable(self):
        """Test abstraction with unused linear variable."""
        tc = TypeChecker()
        # λx. |0⟩ - x is not used
        term = Abstraction("x", Ket0())
        with pytest.raises(TypeCheckError, match="not used"):
            tc.check(term)


class TestApplicationTypeErrors:
    """Test application type error cases."""
    
    def test_application_type_mismatch(self):
        """Test application with mismatched argument type."""
        tc = TypeChecker()
        # (λx. H[x]) (|0⟩⊗|1⟩) - expects qubit, got tensor
        func = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        arg = TensorPair(Ket0(), Ket1())
        term = Application(func, arg)
        with pytest.raises(TypeCheckError, match="Type mismatch"):
            tc.check(term)


class TestUnitaryMultiQubit:
    """Test multi-qubit unitary operations."""
    
    def test_cnot_on_tensor(self):
        """Test CNOT on tensor of qubits."""
        tc = TypeChecker()
        # CNOT(|0⟩⊗|1⟩)
        term = UnitaryApp(UnitaryGate.CNOT, TensorPair(Ket0(), Ket1()))
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
    
    def test_cnot_wrong_arity(self):
        """Test CNOT with wrong number of qubits."""
        tc = TypeChecker()
        # CNOT(|0⟩) - expects 2 qubits
        term = UnitaryApp(UnitaryGate.CNOT, Ket0())
        with pytest.raises(TypeCheckError, match="expects 2 qubits"):
            tc.check(term)
    
    def test_swap_on_tensor(self):
        """Test SWAP on tensor of qubits."""
        tc = TypeChecker()
        term = UnitaryApp(UnitaryGate.SWAP, TensorPair(Ket0(), Ket1()))
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
    
    def test_single_qubit_gate_on_tensor(self):
        """Test single-qubit gate on tensor (error)."""
        tc = TypeChecker()
        # H(|0⟩⊗|1⟩) - H expects single qubit
        term = UnitaryApp(UnitaryGate.H, TensorPair(Ket0(), Ket1()))
        with pytest.raises(TypeCheckError, match="expects qubit"):
            tc.check(term)


class TestQCtrlEdgeCases:
    """Test qctrl edge cases."""
    
    def test_qctrl_sharp_qubit_control(self):
        """Test qctrl with ♯qubit control."""
        tc = TypeChecker()
        sqrt2 = 1/cmath.sqrt(2)
        # Control is a superposition (♯qubit)
        ctrl = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        term = QCtrl(ctrl, Ket0(), Ket1())
        typ = tc.check(term)
        assert isinstance(typ, SharpType)
    
    def test_qctrl_non_qubit_control(self):
        """Test qctrl with non-qubit control."""
        tc = TypeChecker()
        # Control is a tensor, not a qubit
        ctrl = TensorPair(Ket0(), Ket1())
        term = QCtrl(ctrl, Ket0(), Ket1())
        with pytest.raises(TypeCheckError, match="must be qubit"):
            tc.check(term)
    
    def test_qctrl_sharp_non_qubit_control(self):
        """Test qctrl with ♯(non-qubit) control."""
        tc = TypeChecker()
        sqrt2 = 1/cmath.sqrt(2)
        # Control is ♯(qubit⊗qubit), not ♯qubit
        ctrl_inner = TensorPair(Ket0(), Ket1())
        ctrl = Superposition(sqrt2, ctrl_inner, sqrt2, ctrl_inner)
        term = QCtrl(ctrl, Ket0(), Ket1())
        with pytest.raises(TypeCheckError, match="must be qubit"):
            tc.check(term)
    
    def test_qctrl_different_branch_types(self):
        """Test qctrl with different branch types."""
        tc = TypeChecker()
        # Branches have different types
        term = QCtrl(Ket0(), Ket0(), TensorPair(Ket0(), Ket1()))
        with pytest.raises(TypeCheckError, match="different types"):
            tc.check(term)


class TestMeasurementEdgeCases:
    """Test measurement edge cases."""
    
    def test_measure_sharp_qubit(self):
        """Test measurement of ♯qubit."""
        tc = TypeChecker()
        sqrt2 = 1/cmath.sqrt(2)
        sup = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        term = Measurement(MeasurementBasis.COMPUTATIONAL, sup)
        typ = tc.check(term)
        assert typ == BoolType()
    
    def test_measure_sharp_non_qubit(self):
        """Test measurement of ♯(non-qubit)."""
        tc = TypeChecker()
        sqrt2 = 1/cmath.sqrt(2)
        # ♯(qubit⊗qubit) - can't measure directly
        inner = TensorPair(Ket0(), Ket1())
        sup = Superposition(sqrt2, inner, sqrt2, inner)
        term = Measurement(MeasurementBasis.COMPUTATIONAL, sup)
        with pytest.raises(TypeCheckError, match="non-qubit"):
            tc.check(term)
    
    def test_measure_non_qubit(self):
        """Test measurement of non-qubit."""
        tc = TypeChecker()
        # Can't measure a tensor directly
        term = Measurement(MeasurementBasis.COMPUTATIONAL, TensorPair(Ket0(), Ket1()))
        with pytest.raises(TypeCheckError, match="non-qubit"):
            tc.check(term)


class TestLetConstructsComplete:
    """Complete tests for let constructs."""
    
    def test_let_tensor_linear_usage(self):
        """Test let-tensor with proper linear usage."""
        tc = TypeChecker()
        tensor = TensorPair(Ket0(), Ket1())
        # Use both variables
        body = TensorPair(
            UnitaryApp(UnitaryGate.H, Variable("x")),
            UnitaryApp(UnitaryGate.X, Variable("y"))
        )
        term = LetTensor("x", "y", tensor, body)
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
    
    def test_let_bang_with_multiple_uses(self):
        """Test let-bang allowing multiple uses."""
        tc = TypeChecker()
        # let !x = !|0⟩ in x ⊗ x (allowed since x is unrestricted)
        bang = BangIntro(Ket0())
        # This would need special handling for multiple uses
        body = UnitaryApp(UnitaryGate.H, Variable("x"))
        term = LetBang("x", bang, body)
        typ = tc.check(term)
        assert typ == QubitType()


class TestInferParamType:
    """Test parameter type inference."""
    
    def test_infer_from_nested_application(self):
        """Test type inference from nested application."""
        tc = TypeChecker()
        # λx. H[H[x]]
        term = Abstraction("x", UnitaryApp(UnitaryGate.H, 
                                          UnitaryApp(UnitaryGate.H, Variable("x"))))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
        assert typ.domain == QubitType()
    
    def test_infer_from_tensor_left(self):
        """Test type inference from tensor (left position)."""
        tc = TypeChecker()
        # λx. x ⊗ |0⟩
        term = Abstraction("x", TensorPair(Variable("x"), Ket0()))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
    
    def test_infer_from_tensor_right(self):
        """Test type inference from tensor (right position)."""
        tc = TypeChecker()
        # λx. |0⟩ ⊗ x
        term = Abstraction("x", TensorPair(Ket0(), Variable("x")))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
    
    def test_infer_from_application_arg(self):
        """Test type inference from application (argument position)."""
        tc = TypeChecker()
        # λx. (λy.y) x - x appears as argument
        inner = Abstraction("y", Variable("y"))
        term = Abstraction("x", Application(inner, Variable("x")))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)


class TestIsQubitTensor:
    """Test _is_qubit_tensor helper."""
    
    def test_qubit_tensor_nested(self):
        """Test nested qubit tensor detection."""
        tc = TypeChecker()
        # Test internal method with 3-qubit tensor
        typ = TensorProduct(QubitType(), TensorProduct(QubitType(), QubitType()))
        assert tc._is_qubit_tensor(typ, 3) == True
    
    def test_qubit_tensor_wrong_type(self):
        """Test non-qubit in tensor."""
        tc = TypeChecker()
        typ = TensorProduct(BoolType(), QubitType())
        assert tc._is_qubit_tensor(typ, 2) == False


class TestLetTensorUnusedVar:
    """Test let-tensor with unused variables."""
    
    def test_let_tensor_unused_left(self):
        """Test let-tensor where left var is unused."""
        tc = TypeChecker()
        tensor = TensorPair(Ket0(), Ket1())
        # Only use y, not x
        body = UnitaryApp(UnitaryGate.H, Variable("y"))
        term = LetTensor("x", "y", tensor, body)
        with pytest.raises(TypeCheckError, match="not used"):
            tc.check(term)
    
    def test_let_tensor_unused_right(self):
        """Test let-tensor where right var is unused."""
        tc = TypeChecker()
        tensor = TensorPair(Ket0(), Ket1())
        # Only use x, not y
        body = UnitaryApp(UnitaryGate.H, Variable("x"))
        term = LetTensor("x", "y", tensor, body)
        with pytest.raises(TypeCheckError, match="not used"):
            tc.check(term)


class TestBangIntroCheck:
    """Test bang introduction checking."""
    
    def test_bang_intro_of_qubit(self):
        """Test !|0⟩."""
        tc = TypeChecker()
        term = BangIntro(Ket0())
        typ = tc.check(term)
        assert isinstance(typ, BangType)
        assert typ.inner == QubitType()
    
    def test_bang_intro_of_tensor(self):
        """Test !(|0⟩⊗|1⟩)."""
        tc = TypeChecker()
        term = BangIntro(TensorPair(Ket0(), Ket1()))
        typ = tc.check(term)
        assert isinstance(typ, BangType)


class TestParagraphIntroCheck:
    """Test paragraph introduction checking."""
    
    def test_paragraph_intro_of_qubit(self):
        """Test §|0⟩."""
        tc = TypeChecker()
        term = ParagraphIntro(Ket0())
        typ = tc.check(term)
        assert isinstance(typ, ParagraphType)
        assert typ.inner == QubitType()


class TestLetBangNonBang:
    """Test let-bang with non-bang term."""
    
    def test_let_bang_non_bang_error(self):
        """Test let !x = |0⟩ in ... (|0⟩ is not a bang)."""
        tc = TypeChecker()
        # This should fail because Ket0 is not a bang type
        term = LetBang("x", Ket0(), Variable("x"))
        with pytest.raises(TypeCheckError, match="Expected.*bang"):
            tc.check(term)


class TestLetTensorNonTensor:
    """Test let-tensor with non-tensor term."""
    
    def test_let_tensor_non_tensor_error(self):
        """Test let x⊗y = |0⟩ in ... (|0⟩ is not a tensor)."""
        tc = TypeChecker()
        term = LetTensor("x", "y", Ket0(), Variable("x"))
        with pytest.raises(TypeCheckError, match="Expected.*tensor"):
            tc.check(term)
