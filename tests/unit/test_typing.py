"""
Unit tests for λ^QLLR type checking and orthogonality.
"""

import pytest
from qllr.core.syntax import (
    Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit, Inl, Inr
)
from qllr.core.types import (
    QubitType, UnitType, BoolType, LinearArrow, TensorProduct,
    SumType, BangType, ParagraphType, SharpType
)
from qllr.typing.typechecker import TypeChecker, TypeCheckError, Context
from qllr.typing.orthogonality import OrthogonalityChecker, check_orthogonality
import cmath


class TestContext:
    """Tests for typing context."""
    
    def test_empty_context(self):
        ctx = Context()
        assert ctx.is_empty_linear()
        assert ctx.linear_vars() == set()
    
    def test_add_linear(self):
        ctx = Context()
        ctx2 = ctx.add_linear("x", QubitType())
        assert "x" in ctx2.linear
        assert not ctx2.is_empty_linear()
    
    def test_add_unrestricted(self):
        ctx = Context()
        ctx2 = ctx.add_unrestricted("x", QubitType())
        assert "x" in ctx2.unrestricted
        assert ctx2.is_empty_linear()
    
    def test_lookup_linear(self):
        ctx = Context().add_linear("x", QubitType())
        typ, is_linear = ctx.lookup("x")
        assert typ == QubitType()
        assert is_linear == True
    
    def test_lookup_unrestricted(self):
        ctx = Context().add_unrestricted("x", QubitType())
        typ, is_linear = ctx.lookup("x")
        assert typ == QubitType()
        assert is_linear == False
    
    def test_lookup_not_found(self):
        ctx = Context()
        with pytest.raises(TypeCheckError):
            ctx.lookup("x")
    
    def test_remove_linear(self):
        ctx = Context().add_linear("x", QubitType())
        ctx2 = ctx.remove_linear("x")
        assert ctx2.is_empty_linear()
    
    def test_remove_linear_not_found(self):
        ctx = Context()
        with pytest.raises(TypeCheckError):
            ctx.remove_linear("x")
    
    def test_split_context(self):
        ctx = Context()
        ctx = ctx.add_linear("x", QubitType())
        ctx = ctx.add_linear("y", QubitType())
        ctx1, ctx2 = ctx.split({"x"})
        assert "x" in ctx1.linear
        assert "y" in ctx2.linear
    
    def test_merge_context(self):
        ctx1 = Context().add_linear("x", QubitType())
        ctx2 = Context().add_linear("y", QubitType())
        merged = ctx1.merge(ctx2)
        assert "x" in merged.linear
        assert "y" in merged.linear
    
    def test_merge_overlap_error(self):
        ctx1 = Context().add_linear("x", QubitType())
        ctx2 = Context().add_linear("x", QubitType())
        with pytest.raises(TypeCheckError):
            ctx1.merge(ctx2)
    
    def test_context_str(self):
        ctx = Context().add_linear("x", QubitType())
        s = str(ctx)
        assert "x" in s


class TestTypeChecker:
    """Tests for the type checker."""
    
    def test_check_ket0(self):
        tc = TypeChecker()
        typ = tc.check(Ket0())
        assert typ == QubitType()
    
    def test_check_ket1(self):
        tc = TypeChecker()
        typ = tc.check(Ket1())
        assert typ == QubitType()
    
    def test_check_new(self):
        tc = TypeChecker()
        typ = tc.check(New())
        assert typ == QubitType()
    
    def test_check_unit(self):
        tc = TypeChecker()
        typ = tc.check(Unit())
        assert typ == UnitType()
    
    def test_check_variable_in_context(self):
        tc = TypeChecker()
        ctx = Context().add_linear("x", QubitType())
        typ = tc.check(Variable("x"), ctx)
        assert typ == QubitType()
    
    def test_check_variable_not_in_context(self):
        tc = TypeChecker()
        with pytest.raises(TypeCheckError):
            tc.check(Variable("x"))
    
    def test_check_hadamard(self):
        tc = TypeChecker()
        term = UnitaryApp(UnitaryGate.H, Ket0())
        typ = tc.check(term)
        assert typ == QubitType()
    
    def test_check_pauli_x(self):
        tc = TypeChecker()
        term = UnitaryApp(UnitaryGate.X, Ket1())
        typ = tc.check(term)
        assert typ == QubitType()
    
    def test_check_tensor_pair(self):
        tc = TypeChecker()
        term = TensorPair(Ket0(), Ket1())
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
        assert typ.left == QubitType()
        assert typ.right == QubitType()
    
    def test_check_abstraction_identity(self):
        tc = TypeChecker()
        # λx. H[x] : qubit ⊸ qubit
        term = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
        assert typ.domain == QubitType()
        assert typ.codomain == QubitType()
    
    def test_check_application(self):
        tc = TypeChecker()
        # (λx. H[x]) |0⟩
        func = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        term = Application(func, Ket0())
        typ = tc.check(term)
        assert typ == QubitType()
    
    def test_check_bang_intro(self):
        tc = TypeChecker()
        term = BangIntro(Ket0())
        typ = tc.check(term)
        assert isinstance(typ, BangType)
        assert typ.inner == QubitType()
    
    def test_check_let_bang(self):
        tc = TypeChecker()
        # let !x = !|0⟩ in H[x]
        term = LetBang("x", BangIntro(Ket0()), 
                       UnitaryApp(UnitaryGate.H, Variable("x")))
        typ = tc.check(term)
        assert typ == QubitType()
    
    def test_check_paragraph_intro(self):
        tc = TypeChecker()
        term = ParagraphIntro(Ket0())
        typ = tc.check(term)
        assert isinstance(typ, ParagraphType)
    
    def test_check_measurement(self):
        tc = TypeChecker()
        term = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        typ = tc.check(term)
        assert typ == BoolType()
    
    def test_check_qctrl_basic(self):
        tc = TypeChecker()
        # qctrl(|0⟩, |0⟩, |1⟩)
        term = QCtrl(Ket0(), Ket0(), Ket1())
        typ = tc.check(term)
        assert isinstance(typ, SharpType)
    
    def test_check_superposition(self):
        tc = TypeChecker()
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        typ = tc.check(term)
        assert isinstance(typ, SharpType)
    
    def test_check_inl(self):
        tc = TypeChecker()
        term = Inl(Ket0())
        typ = tc.check(term)
        assert isinstance(typ, SumType)
    
    def test_check_inr(self):
        tc = TypeChecker()
        term = Inr(Ket1())
        typ = tc.check(term)
        assert isinstance(typ, SumType)
    
    def test_check_let_tensor(self):
        tc = TypeChecker()
        # let x ⊗ y = (|0⟩ ⊗ |1⟩) in x ⊗ y (use both variables)
        tensor = TensorPair(Ket0(), Ket1())
        term = LetTensor("x", "y", tensor, 
                        TensorPair(Variable("x"), Variable("y")))
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
    
    def test_linear_variable_used_once(self):
        tc = TypeChecker()
        ctx = Context().add_linear("x", QubitType())
        # Using x once is OK
        typ = tc.check(UnitaryApp(UnitaryGate.H, Variable("x")), ctx)
        assert typ == QubitType()
    
    def test_unused_linear_variable_error(self):
        tc = TypeChecker()
        ctx = Context().add_linear("x", QubitType())
        # Not using x at all should error
        with pytest.raises(TypeCheckError):
            tc.check(Ket0(), ctx)


class TestOrthogonalityChecker:
    """Tests for orthogonality checking."""
    
    def test_ket0_ket1_orthogonal(self):
        checker = OrthogonalityChecker()
        assert checker.check_orthogonal(Ket0(), Ket1()) == True
    
    def test_ket1_ket0_orthogonal(self):
        checker = OrthogonalityChecker()
        assert checker.check_orthogonal(Ket1(), Ket0()) == True
    
    def test_ket0_ket0_not_orthogonal(self):
        checker = OrthogonalityChecker()
        assert checker.check_orthogonal(Ket0(), Ket0()) == False
    
    def test_ket1_ket1_not_orthogonal(self):
        checker = OrthogonalityChecker()
        assert checker.check_orthogonal(Ket1(), Ket1()) == False
    
    def test_tensor_orthogonal_left(self):
        checker = OrthogonalityChecker()
        t0 = TensorPair(Ket0(), Ket0())
        t1 = TensorPair(Ket1(), Ket0())
        assert checker.check_orthogonal(t0, t1) == True
    
    def test_tensor_orthogonal_right(self):
        checker = OrthogonalityChecker()
        t0 = TensorPair(Ket0(), Ket0())
        t1 = TensorPair(Ket0(), Ket1())
        assert checker.check_orthogonal(t0, t1) == True
    
    def test_unitary_preserves_orthogonality(self):
        checker = OrthogonalityChecker()
        # H|0⟩ and H|1⟩ are orthogonal
        t0 = UnitaryApp(UnitaryGate.H, Ket0())
        t1 = UnitaryApp(UnitaryGate.H, Ket1())
        assert checker.check_orthogonal(t0, t1) == True
    
    def test_higher_order_orthogonal(self):
        checker = OrthogonalityChecker()
        # λx. (H[x] ⊗ |0⟩) and λx. (H[x] ⊗ |1⟩)
        f0 = Abstraction("x", TensorPair(
            UnitaryApp(UnitaryGate.H, Variable("x")),
            Ket0()
        ))
        f1 = Abstraction("x", TensorPair(
            UnitaryApp(UnitaryGate.H, Variable("x")),
            Ket1()
        ))
        assert checker.check_orthogonal(f0, f1) == True
    
    def test_higher_order_not_orthogonal_depends_on_var(self):
        checker = OrthogonalityChecker()
        # λx. x and λx. H[x] - difference depends on x
        f0 = Abstraction("x", Variable("x"))
        f1 = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        # This should be False because the difference involves x
        assert checker.check_orthogonal(f0, f1) == False
    
    def test_check_orthogonality_convenience(self):
        assert check_orthogonality(Ket0(), Ket1()) == True
        assert check_orthogonality(Ket0(), Ket0()) == False
    
    def test_find_differences_empty(self):
        checker = OrthogonalityChecker()
        diffs = checker._find_differences(Ket0(), Ket0(), [])
        assert diffs == []
    
    def test_find_differences_single(self):
        checker = OrthogonalityChecker()
        diffs = checker._find_differences(Ket0(), Ket1(), [])
        assert len(diffs) == 1
    
    def test_structurally_equal(self):
        checker = OrthogonalityChecker()
        assert checker._structurally_equal(Ket0(), Ket0()) == True
        assert checker._structurally_equal(Ket0(), Ket1()) == False
    
    def test_difference_involves_bound_var(self):
        from qllr.typing.orthogonality import Difference
        diff = Difference([], Variable("x"), Variable("y"))
        assert diff.involves_bound_var({"x"}) == True
        assert diff.involves_bound_var({"z"}) == False


class TestTypeCheckerWithOrthogonality:
    """Tests for type checker with orthogonality checking."""
    
    def test_qctrl_with_orthogonality_checker(self):
        orth_checker = OrthogonalityChecker()
        tc = TypeChecker(orthogonality_checker=orth_checker)
        
        # qctrl(|0⟩, |0⟩, |1⟩) - branches are orthogonal
        term = QCtrl(Ket0(), Ket0(), Ket1())
        typ = tc.check(term)
        assert isinstance(typ, SharpType)
    
    def test_qctrl_non_orthogonal_branches(self):
        orth_checker = OrthogonalityChecker()
        tc = TypeChecker(orthogonality_checker=orth_checker)
        
        # qctrl(|0⟩, |0⟩, |0⟩) - branches are NOT orthogonal
        term = QCtrl(Ket0(), Ket0(), Ket0())
        with pytest.raises(TypeCheckError):
            tc.check(term)


class TestComplexTyping:
    """Tests for complex typing scenarios."""
    
    def test_nested_abstraction(self):
        tc = TypeChecker()
        # λx. λy. x ⊗ y (simpler nested abstraction)
        term = Abstraction("x", Abstraction("y", 
            TensorPair(Variable("x"), Variable("y"))))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
    
    def test_bell_state_preparation(self):
        tc = TypeChecker()
        # H[|0⟩] ⊗ |0⟩ followed by CNOT would give Bell state
        term = TensorPair(UnitaryApp(UnitaryGate.H, Ket0()), Ket0())
        typ = tc.check(term)
        assert isinstance(typ, TensorProduct)
    
    def test_quantum_teleportation_fragment(self):
        tc = TypeChecker()
        # Just the measurement part: meas(H[|0⟩])
        term = Measurement(MeasurementBasis.COMPUTATIONAL,
                          UnitaryApp(UnitaryGate.H, Ket0()))
        typ = tc.check(term)
        assert typ == BoolType()


class TestTypeInference:
    """Tests for type inference in abstractions."""
    
    def test_infer_qubit_from_unitary(self):
        tc = TypeChecker()
        # λx. H[x] should infer x : qubit
        term = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
        assert typ.domain == QubitType()
    
    def test_infer_from_tensor(self):
        tc = TypeChecker()
        # λx. x ⊗ |0⟩
        term = Abstraction("x", TensorPair(Variable("x"), Ket0()))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
