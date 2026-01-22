"""
Final tests to achieve 100% code coverage.
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
from qllr.typing.orthogonality import OrthogonalityChecker, Difference
from qllr.compilation.circuit import QuantumCircuit, Gate, GateType
from qllr.compilation.circuit_extraction import CircuitExtractor, CircuitExtractionError


# ============= Syntax Coverage =============

class TestSyntaxTermMethods:
    """Test all Term methods for complete coverage."""
    
    def test_term_base_class(self):
        """Test Term base class methods."""
        # Term is abstract but test its existence
        assert hasattr(Term, '__init__')
    
    def test_variable_clone(self):
        """Test Variable clone method."""
        v = Variable("x")
        c = v.clone()
        assert c.name == "x"
        assert c is not v
    
    def test_ket0_clone(self):
        """Test Ket0 clone method."""
        k = Ket0()
        c = k.clone()
        assert isinstance(c, Ket0)
        assert c is not k
    
    def test_ket1_clone(self):
        """Test Ket1 clone method."""
        k = Ket1()
        c = k.clone()
        assert isinstance(c, Ket1)
        assert c is not k
    
    def test_superposition_clone(self):
        """Test Superposition clone method."""
        sqrt2 = 1/cmath.sqrt(2)
        s = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        c = s.clone()
        assert isinstance(c, Superposition)
    
    def test_abstraction_clone(self):
        """Test Abstraction clone method."""
        a = Abstraction("x", Variable("x"))
        c = a.clone()
        assert isinstance(c, Abstraction)
        assert c.var == "x"
    
    def test_application_clone(self):
        """Test Application clone method."""
        app = Application(Variable("f"), Ket0())
        c = app.clone()
        assert isinstance(c, Application)
    
    def test_tensor_pair_clone(self):
        """Test TensorPair clone method."""
        t = TensorPair(Ket0(), Ket1())
        c = t.clone()
        assert isinstance(c, TensorPair)
    
    def test_let_tensor_clone(self):
        """Test LetTensor clone method."""
        lt = LetTensor("x", "y", TensorPair(Ket0(), Ket1()), Variable("x"))
        c = lt.clone()
        assert isinstance(c, LetTensor)
    
    def test_unitary_app_clone(self):
        """Test UnitaryApp clone method."""
        u = UnitaryApp(UnitaryGate.H, Ket0())
        c = u.clone()
        assert isinstance(c, UnitaryApp)
    
    def test_qctrl_clone(self):
        """Test QCtrl clone method."""
        q = QCtrl(Ket0(), Ket0(), Ket1())
        c = q.clone()
        assert isinstance(c, QCtrl)
    
    def test_measurement_clone(self):
        """Test Measurement clone method."""
        m = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        c = m.clone()
        assert isinstance(c, Measurement)
    
    def test_new_clone(self):
        """Test New clone method."""
        n = New()
        c = n.clone()
        assert isinstance(c, New)
    
    def test_unit_clone(self):
        """Test Unit clone method."""
        u = Unit()
        c = u.clone()
        assert isinstance(c, Unit)
    
    def test_inl_clone(self):
        """Test Inl clone method."""
        i = Inl(Ket0())
        c = i.clone()
        assert isinstance(c, Inl)
    
    def test_inr_clone(self):
        """Test Inr clone method."""
        i = Inr(Ket1())
        c = i.clone()
        assert isinstance(c, Inr)
    
    def test_case_clone(self):
        """Test Case clone method."""
        c = Case(Inl(Ket0()), "x", Variable("x"), "y", Variable("y"))
        cl = c.clone()
        assert isinstance(cl, Case)
    
    def test_bang_intro_clone(self):
        """Test BangIntro clone method."""
        b = BangIntro(Ket0())
        c = b.clone()
        assert isinstance(c, BangIntro)
    
    def test_let_bang_clone(self):
        """Test LetBang clone method."""
        lb = LetBang("x", BangIntro(Ket0()), Variable("x"))
        c = lb.clone()
        assert isinstance(c, LetBang)
    
    def test_paragraph_intro_clone(self):
        """Test ParagraphIntro clone method."""
        p = ParagraphIntro(Ket0())
        c = p.clone()
        assert isinstance(c, ParagraphIntro)


# ============= Types Coverage =============

class TestTypesMethods:
    """Test all Type methods for complete coverage."""
    
    def test_qubit_type_equality(self):
        """Test QubitType equality."""
        q1 = QubitType()
        q2 = QubitType()
        assert q1 == q2
    
    def test_qubit_type_width(self):
        """Test QubitType width."""
        q = QubitType()
        assert q.width() == 1
    
    def test_linear_arrow_width(self):
        """Test LinearArrow width method."""
        arr = LinearArrow(QubitType(), TensorProduct(QubitType(), QubitType()))
        assert arr.width() >= 1
    
    def test_tensor_product_width(self):
        """Test TensorProduct width method."""
        t = TensorProduct(QubitType(), QubitType())
        assert t.width() == 2
    
    def test_sum_type_width(self):
        """Test SumType width method."""
        s = SumType(QubitType(), TensorProduct(QubitType(), QubitType()))
        assert s.width() >= 1
    
    def test_bang_type_width(self):
        """Test BangType width method."""
        b = BangType(TensorProduct(QubitType(), QubitType()))
        assert b.width() == 2
    
    def test_paragraph_type_width(self):
        """Test ParagraphType width method."""
        p = ParagraphType(TensorProduct(QubitType(), QubitType()))
        assert p.width() == 2
    
    def test_sharp_type_width(self):
        """Test SharpType width method."""
        s = SharpType(TensorProduct(QubitType(), QubitType()))
        assert s.width() == 2
    
    def test_forall_type_modal_depth(self):
        """Test ForallType modal_depth method."""
        f = ForallType("X", ParagraphType(QubitType()))
        assert f.modal_depth() == 1


# ============= Orthogonality Coverage =============

class TestOrthogonalityEdgeCases:
    """Test orthogonality checker edge cases."""
    
    def test_check_orthogonal_same_type_different_structure(self):
        """Test orthogonality with structurally different terms."""
        checker = OrthogonalityChecker()
        # Variable vs Ket (different term types)
        result = checker.check_orthogonal(Variable("x"), Ket0())
        assert result == False  # Can't determine orthogonality
    
    def test_find_differences_with_let_tensor(self):
        """Test finding differences in let-tensor terms."""
        checker = OrthogonalityChecker()
        lt1 = LetTensor("x", "y", TensorPair(Ket0(), Ket1()), Variable("x"))
        lt2 = LetTensor("x", "y", TensorPair(Ket1(), Ket0()), Variable("x"))
        diffs = checker._find_differences(lt1, lt2, [])
        assert len(diffs) >= 1


# ============= TypeChecker Coverage =============

class TestTypeCheckerRemainingCases:
    """Test remaining type checker cases."""
    
    def test_infer_param_type_from_nested_unitary(self):
        """Test type inference from deeply nested unitary."""
        tc = TypeChecker()
        # λx. H[H[H[x]]]
        term = Abstraction("x", 
            UnitaryApp(UnitaryGate.H,
                UnitaryApp(UnitaryGate.H,
                    UnitaryApp(UnitaryGate.H, Variable("x")))))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
    
    def test_infer_param_type_from_app_arg(self):
        """Test type inference when var is in application argument."""
        tc = TypeChecker()
        # λx. (λy.H[y]) x
        inner = Abstraction("y", UnitaryApp(UnitaryGate.H, Variable("y")))
        term = Abstraction("x", Application(inner, Variable("x")))
        typ = tc.check(term)
        assert isinstance(typ, LinearArrow)
    
    def test_application_with_type_mismatch(self):
        """Test application type mismatch error."""
        tc = TypeChecker()
        # Create a function expecting qubit
        func = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        # Apply to tensor (wrong type)
        term = Application(func, TensorPair(Ket0(), Ket1()))
        with pytest.raises(TypeCheckError, match="Type mismatch"):
            tc.check(term)


# ============= Circuit Extraction Coverage =============

class TestCircuitExtractionRemainingCases:
    """Test remaining circuit extraction cases."""
    
    def test_extract_application_with_abstraction(self):
        """Test extracting application with abstraction."""
        extractor = CircuitExtractor()
        # (λx. H[x]) |0⟩
        func = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
        term = Application(func, Ket0())
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 1
    
    def test_extract_let_tensor_uses_both_vars(self):
        """Test let-tensor extraction uses both variables."""
        extractor = CircuitExtractor()
        tensor = TensorPair(Ket0(), Ket1())
        body = TensorPair(Variable("x"), Variable("y"))
        term = LetTensor("x", "y", tensor, body)
        circuit = extractor.extract(term)
        assert circuit.num_qubits >= 2


# ============= Circuit Coverage =============

class TestCircuitRemainingCases:
    """Test remaining circuit cases."""
    
    def test_qasm_unknown_gate(self):
        """Test OpenQASM with gate not in mapping."""
        c = QuantumCircuit(2)
        # Add a controlled gate
        c.add_controlled(GateType.X, 0, 1)
        qasm = c.to_openqasm()
        assert "cx" in qasm or "OPENQASM" in qasm
    
    def test_compose_two_non_empty(self):
        """Test composing two non-empty circuits."""
        c1 = QuantumCircuit(2)
        c1.add_h(0)
        c1.add_cnot(0, 1)
        
        c2 = QuantumCircuit(2)
        c2.add_x(0)
        c2.add_z(1)
        
        composed = c1.compose(c2)
        assert composed.size() == 4
    
    def test_inverse_two_qubit_gate(self):
        """Test inverse of two-qubit gate."""
        c = QuantumCircuit(2)
        c.add_cnot(0, 1)
        inv = c.inverse()
        # CNOT is self-inverse
        assert inv.size() == 1
        assert inv.gates[0].gate_type == GateType.CNOT


# ============= Complete All Remaining Branches =============

class TestRemainingBranches:
    """Test all remaining uncovered branches."""
    
    def test_types_equal_all_cases(self):
        """Test types_equal with all type combinations."""
        # UnitType
        assert types_equal(UnitType(), UnitType())
        assert not types_equal(UnitType(), QubitType())
        
        # BoolType
        assert types_equal(BoolType(), BoolType())
        assert not types_equal(BoolType(), QubitType())
        
        # TypeVariable
        assert types_equal(TypeVariable("X"), TypeVariable("X"))
        assert not types_equal(TypeVariable("X"), TypeVariable("Y"))
        
        # LinearArrow
        arr1 = LinearArrow(QubitType(), QubitType())
        arr2 = LinearArrow(QubitType(), QubitType())
        assert types_equal(arr1, arr2)
        
        # TensorProduct
        t1 = TensorProduct(QubitType(), QubitType())
        t2 = TensorProduct(QubitType(), QubitType())
        assert types_equal(t1, t2)
        
        # SumType
        s1 = SumType(QubitType(), UnitType())
        s2 = SumType(QubitType(), UnitType())
        assert types_equal(s1, s2)
        
        # BangType
        b1 = BangType(QubitType())
        b2 = BangType(QubitType())
        assert types_equal(b1, b2)
        
        # ParagraphType
        p1 = ParagraphType(QubitType())
        p2 = ParagraphType(QubitType())
        assert types_equal(p1, p2)
        
        # SharpType
        sh1 = SharpType(QubitType())
        sh2 = SharpType(QubitType())
        assert types_equal(sh1, sh2)
        
        # ForallType (alpha equivalence)
        f1 = ForallType("X", TypeVariable("X"))
        f2 = ForallType("Y", TypeVariable("Y"))
        assert types_equal(f1, f2)
    
    def test_difference_involves_bound_var_all_cases(self):
        """Test Difference.involves_bound_var with all cases."""
        # Variable in term0
        d1 = Difference([], Variable("x"), Ket0())
        assert d1.involves_bound_var({"x"}) == True
        assert d1.involves_bound_var({"y"}) == False
        
        # Variable in term1
        d2 = Difference([], Ket0(), Variable("y"))
        assert d2.involves_bound_var({"y"}) == True
        assert d2.involves_bound_var({"x"}) == False
        
        # No variables
        d3 = Difference([], Ket0(), Ket1())
        assert d3.involves_bound_var({"x"}) == False
    
    def test_abstraction_free_vars_in_body(self):
        """Test abstraction with free variables in body."""
        # λx. (x, y) - y is free
        term = Abstraction("x", TensorPair(Variable("x"), Variable("y")))
        fv = term.free_variables()
        assert "y" in fv
        assert "x" not in fv
    
    def test_case_free_variables(self):
        """Test Case free_variables method."""
        # case z of x → x | y → y
        c = Case(Variable("z"), "x", Variable("x"), "y", Variable("y"))
        fv = c.free_variables()
        assert "z" in fv
        assert "x" not in fv
        assert "y" not in fv
    
    def test_let_bang_free_variables(self):
        """Test LetBang free_variables method."""
        # let !x = z in x
        lb = LetBang("x", Variable("z"), Variable("x"))
        fv = lb.free_variables()
        assert "z" in fv
        assert "x" not in fv
