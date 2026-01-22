"""
Unit tests for λ^QLLR core syntax and types.
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


class TestBasisStates:
    """Tests for basis state terms."""
    
    def test_ket0_creation(self):
        k0 = Ket0()
        assert isinstance(k0, Ket0)
        assert str(k0) == "|0⟩"
    
    def test_ket1_creation(self):
        k1 = Ket1()
        assert isinstance(k1, Ket1)
        assert str(k1) == "|1⟩"
    
    def test_ket0_is_value(self):
        assert Ket0().is_value() == True
    
    def test_ket1_is_value(self):
        assert Ket1().is_value() == True
    
    def test_ket0_free_variables(self):
        assert Ket0().free_variables() == set()
    
    def test_ket1_free_variables(self):
        assert Ket1().free_variables() == set()
    
    def test_ket0_substitution(self):
        k0 = Ket0()
        result = k0.substitute("x", Ket1())
        assert result == k0
    
    def test_ket0_equality(self):
        assert Ket0() == Ket0()
        assert Ket0() != Ket1()
    
    def test_ket0_hash(self):
        assert hash(Ket0()) == hash(Ket0())


class TestVariables:
    """Tests for variable terms."""
    
    def test_variable_creation(self):
        x = Variable("x")
        assert x.name == "x"
        assert str(x) == "x"
    
    def test_variable_is_value(self):
        assert Variable("x").is_value() == True
    
    def test_variable_free_variables(self):
        x = Variable("x")
        assert x.free_variables() == {"x"}
    
    def test_variable_substitution_match(self):
        x = Variable("x")
        result = x.substitute("x", Ket0())
        assert result == Ket0()
    
    def test_variable_substitution_no_match(self):
        x = Variable("x")
        result = x.substitute("y", Ket0())
        assert result == x
    
    def test_variable_equality(self):
        assert Variable("x") == Variable("x")
        assert Variable("x") != Variable("y")


class TestSuperposition:
    """Tests for superposition terms."""
    
    def test_superposition_creation(self):
        sqrt2 = 1/cmath.sqrt(2)
        sup = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        assert sup.alpha == sqrt2
        assert sup.beta == sqrt2
    
    def test_superposition_is_normalized(self):
        sqrt2 = 1/cmath.sqrt(2)
        sup = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        assert sup.is_normalized() == True
    
    def test_superposition_unnormalized(self):
        sup = Superposition(0.5, Ket0(), 0.5, Ket1())
        assert sup.is_normalized() == False
    
    def test_superposition_free_variables(self):
        sup = Superposition(0.5, Variable("x"), 0.5, Variable("y"))
        assert sup.free_variables() == {"x", "y"}
    
    def test_superposition_substitution(self):
        sqrt2 = 1/cmath.sqrt(2)
        sup = Superposition(sqrt2, Variable("x"), sqrt2, Ket1())
        result = sup.substitute("x", Ket0())
        assert isinstance(result, Superposition)
        assert result.term0 == Ket0()
    
    def test_superposition_is_value(self):
        sqrt2 = 1/cmath.sqrt(2)
        sup = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        assert sup.is_value() == True


class TestAbstraction:
    """Tests for lambda abstraction terms."""
    
    def test_abstraction_creation(self):
        body = Variable("x")
        lam = Abstraction("x", body)
        assert lam.var == "x"
        assert lam.body == body
    
    def test_abstraction_is_value(self):
        assert Abstraction("x", Variable("x")).is_value() == True
    
    def test_abstraction_free_variables(self):
        lam = Abstraction("x", Variable("x"))
        assert lam.free_variables() == set()
        
        lam2 = Abstraction("x", Variable("y"))
        assert lam2.free_variables() == {"y"}
    
    def test_abstraction_substitution_no_capture(self):
        lam = Abstraction("x", Variable("y"))
        result = lam.substitute("y", Variable("z"))
        assert isinstance(result, Abstraction)
        assert result.body == Variable("z")
    
    def test_abstraction_substitution_shadowed(self):
        lam = Abstraction("x", Variable("x"))
        result = lam.substitute("x", Ket0())
        # Variable is shadowed, no substitution in body
        assert result.body == Variable("x")


class TestApplication:
    """Tests for application terms."""
    
    def test_application_creation(self):
        func = Abstraction("x", Variable("x"))
        arg = Ket0()
        app = Application(func, arg)
        assert app.func == func
        assert app.arg == arg
    
    def test_application_is_not_value(self):
        app = Application(Abstraction("x", Variable("x")), Ket0())
        assert app.is_value() == False
    
    def test_application_free_variables(self):
        app = Application(Variable("f"), Variable("x"))
        assert app.free_variables() == {"f", "x"}
    
    def test_application_substitution(self):
        app = Application(Variable("f"), Variable("x"))
        result = app.substitute("x", Ket0())
        assert isinstance(result, Application)
        assert result.arg == Ket0()


class TestTensorPair:
    """Tests for tensor product terms."""
    
    def test_tensor_pair_creation(self):
        pair = TensorPair(Ket0(), Ket1())
        assert pair.left == Ket0()
        assert pair.right == Ket1()
    
    def test_tensor_pair_is_value(self):
        pair = TensorPair(Ket0(), Ket1())
        assert pair.is_value() == True
    
    def test_tensor_pair_free_variables(self):
        pair = TensorPair(Variable("x"), Variable("y"))
        assert pair.free_variables() == {"x", "y"}


class TestUnitaryApp:
    """Tests for unitary gate applications."""
    
    def test_hadamard_creation(self):
        h = UnitaryApp(UnitaryGate.H, Ket0())
        assert h.gate == UnitaryGate.H
        assert h.arg == Ket0()
    
    def test_pauli_gates(self):
        x = UnitaryApp(UnitaryGate.X, Ket0())
        y = UnitaryApp(UnitaryGate.Y, Ket0())
        z = UnitaryApp(UnitaryGate.Z, Ket0())
        assert x.gate == UnitaryGate.X
        assert y.gate == UnitaryGate.Y
        assert z.gate == UnitaryGate.Z
    
    def test_gate_arity(self):
        assert UnitaryGate.H.arity() == 1
        assert UnitaryGate.CNOT.arity() == 2
    
    def test_unitary_free_variables(self):
        h = UnitaryApp(UnitaryGate.H, Variable("q"))
        assert h.free_variables() == {"q"}


class TestQCtrl:
    """Tests for quantum control terms."""
    
    def test_qctrl_creation(self):
        ctrl = QCtrl(Ket0(), Ket0(), Ket1())
        assert ctrl.control == Ket0()
        assert ctrl.branch0 == Ket0()
        assert ctrl.branch1 == Ket1()
    
    def test_qctrl_free_variables(self):
        ctrl = QCtrl(Variable("c"), Variable("a"), Variable("b"))
        assert ctrl.free_variables() == {"c", "a", "b"}
    
    def test_qctrl_substitution(self):
        ctrl = QCtrl(Variable("c"), Ket0(), Ket1())
        result = ctrl.substitute("c", Ket0())
        assert result.control == Ket0()


class TestMeasurement:
    """Tests for measurement terms."""
    
    def test_measurement_creation(self):
        m = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        assert m.basis == MeasurementBasis.COMPUTATIONAL
        assert m.arg == Ket0()
    
    def test_measurement_bases(self):
        assert MeasurementBasis.COMPUTATIONAL.value == "Z"
        assert MeasurementBasis.HADAMARD.value == "X"


class TestTypes:
    """Tests for type definitions."""
    
    def test_qubit_type(self):
        q = QubitType()
        assert str(q) == "qubit"
        assert q.is_linear() == True
        assert q.modal_depth() == 0
    
    def test_linear_arrow(self):
        arr = LinearArrow(QubitType(), QubitType())
        assert arr.is_linear() == True
        assert arr.modal_depth() == 0
    
    def test_tensor_product(self):
        tensor = TensorProduct(QubitType(), QubitType())
        assert tensor.is_linear() == True
        assert tensor.modal_depth() == 0
    
    def test_bang_type(self):
        bang = BangType(QubitType())
        assert bang.is_linear() == False
        assert bang.modal_depth() == 0
    
    def test_paragraph_type(self):
        para = ParagraphType(QubitType())
        assert para.is_linear() == False
        assert para.modal_depth() == 1
    
    def test_nested_paragraph(self):
        para2 = ParagraphType(ParagraphType(QubitType()))
        assert para2.modal_depth() == 2
    
    def test_sharp_type(self):
        sharp = SharpType(QubitType())
        assert sharp.is_linear() == True
    
    def test_forall_type(self):
        forall = ForallType("X", LinearArrow(TypeVariable("X"), TypeVariable("X")))
        assert forall.free_type_variables() == set()
    
    def test_types_equal(self):
        assert types_equal(QubitType(), QubitType())
        assert not types_equal(QubitType(), UnitType())
        assert types_equal(
            LinearArrow(QubitType(), QubitType()),
            LinearArrow(QubitType(), QubitType())
        )


class TestTypeSubstitution:
    """Tests for type substitution."""
    
    def test_type_variable_substitution(self):
        tv = TypeVariable("X")
        result = tv.substitute_type("X", QubitType())
        assert result == QubitType()
    
    def test_type_variable_no_substitution(self):
        tv = TypeVariable("X")
        result = tv.substitute_type("Y", QubitType())
        assert result == tv
    
    def test_linear_arrow_substitution(self):
        arr = LinearArrow(TypeVariable("X"), TypeVariable("X"))
        result = arr.substitute_type("X", QubitType())
        assert types_equal(result, LinearArrow(QubitType(), QubitType()))
    
    def test_forall_instantiation(self):
        forall = ForallType("X", TypeVariable("X"))
        result = forall.instantiate(QubitType())
        assert result == QubitType()


class TestTermClone:
    """Tests for term cloning."""
    
    def test_clone_variable(self):
        x = Variable("x")
        clone = x.clone()
        assert clone == x
        assert clone is not x
    
    def test_clone_abstraction(self):
        lam = Abstraction("x", Variable("x"))
        clone = lam.clone()
        assert clone == lam
        assert clone is not lam


class TestLetConstructs:
    """Tests for let bindings."""
    
    def test_let_tensor(self):
        lt = LetTensor("x", "y", TensorPair(Ket0(), Ket1()), Variable("x"))
        assert lt.var_left == "x"
        assert lt.var_right == "y"
    
    def test_let_bang(self):
        lb = LetBang("x", BangIntro(Ket0()), Variable("x"))
        assert lb.var == "x"
    
    def test_let_tensor_free_variables(self):
        lt = LetTensor("x", "y", Variable("z"), 
                       Application(Variable("x"), Variable("y")))
        assert lt.free_variables() == {"z"}


class TestModalities:
    """Tests for modal terms."""
    
    def test_bang_intro(self):
        bang = BangIntro(Ket0())
        assert bang.term == Ket0()
        assert bang.is_value() == True
    
    def test_paragraph_intro(self):
        para = ParagraphIntro(Ket0())
        assert para.term == Ket0()
        assert para.is_value() == True


class TestSumTypes:
    """Tests for sum type terms."""
    
    def test_inl(self):
        inl = Inl(Ket0())
        assert inl.term == Ket0()
        assert inl.is_value() == True
    
    def test_inr(self):
        inr = Inr(Ket1())
        assert inr.term == Ket1()
        assert inr.is_value() == True
    
    def test_case(self):
        case = Case(
            Inl(Ket0()),
            "x", Variable("x"),
            "y", Variable("y")
        )
        assert case.var_left == "x"
        assert case.var_right == "y"


class TestUnitaryMatrices:
    """Tests for unitary gate matrices."""
    
    def test_hadamard_matrix(self):
        h_matrix = UnitaryGate.H.matrix()
        assert len(h_matrix) == 2
        assert len(h_matrix[0]) == 2
        # Check unitary property: H^2 = I (approximately)
        sqrt2 = 1/cmath.sqrt(2)
        assert abs(h_matrix[0][0] - sqrt2) < 1e-10
    
    def test_pauli_x_matrix(self):
        x_matrix = UnitaryGate.X.matrix()
        assert x_matrix[0][1] == 1
        assert x_matrix[1][0] == 1
    
    def test_cnot_matrix(self):
        cnot_matrix = UnitaryGate.CNOT.matrix()
        assert len(cnot_matrix) == 4


class TestNewAndUnit:
    """Tests for new and unit terms."""
    
    def test_new(self):
        n = New()
        assert n.is_value() == True
        assert n.free_variables() == set()
    
    def test_unit(self):
        u = Unit()
        assert u.is_value() == True
        assert u.free_variables() == set()
