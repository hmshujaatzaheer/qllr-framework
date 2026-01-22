"""
Additional tests for 100% coverage on syntax module.
"""

import pytest
import cmath
from qllr.core.syntax import (
    Term, Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit, Inl, Inr, Case
)


class TestKet0Complete:
    """Complete tests for Ket0."""
    
    def test_repr(self):
        k = Ket0()
        assert repr(k) == "Ket0()"


class TestKet1Complete:
    """Complete tests for Ket1."""
    
    def test_repr(self):
        k = Ket1()
        assert repr(k) == "Ket1()"
    
    def test_substitute(self):
        k = Ket1()
        result = k.substitute("x", Ket0())
        assert result == k


class TestVariableComplete:
    """Complete tests for Variable."""
    
    def test_repr(self):
        v = Variable("x")
        assert repr(v) == "Variable('x')"


class TestSuperpositionComplete:
    """Complete tests for Superposition."""
    
    def test_str(self):
        sqrt2 = 1/cmath.sqrt(2)
        s = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        assert "+" in str(s)
    
    def test_repr(self):
        s = Superposition(0.5, Ket0(), 0.5, Ket1())
        assert "Superposition" in repr(s)
    
    def test_is_value_with_non_value(self):
        # Superposition with non-value terms
        app = Application(Abstraction("x", Variable("x")), Ket0())
        s = Superposition(0.5, app, 0.5, Ket1())
        assert s.is_value() == False


class TestAbstractionComplete:
    """Complete tests for Abstraction."""
    
    def test_str(self):
        lam = Abstraction("x", Variable("x"))
        assert "λ" in str(lam)
    
    def test_repr(self):
        lam = Abstraction("x", Variable("x"))
        assert "Abstraction" in repr(lam)
    
    def test_substitution_with_capture_avoidance(self):
        # λx. y where we substitute y := x (needs alpha conversion)
        lam = Abstraction("x", Variable("y"))
        result = lam.substitute("y", Variable("x"))
        assert isinstance(result, Abstraction)
        # Variable name should be different to avoid capture
        assert result.var != "x" or result.body != Variable("x")


class TestApplicationComplete:
    """Complete tests for Application."""
    
    def test_str(self):
        app = Application(Variable("f"), Variable("x"))
        assert "f" in str(app)
    
    def test_repr(self):
        app = Application(Variable("f"), Variable("x"))
        assert "Application" in repr(app)


class TestTensorPairComplete:
    """Complete tests for TensorPair."""
    
    def test_str(self):
        t = TensorPair(Ket0(), Ket1())
        assert "⊗" in str(t)
    
    def test_repr(self):
        t = TensorPair(Ket0(), Ket1())
        assert "TensorPair" in repr(t)
    
    def test_substitution(self):
        t = TensorPair(Variable("x"), Variable("y"))
        result = t.substitute("x", Ket0())
        assert result.left == Ket0()
    
    def test_is_value_with_non_value(self):
        app = Application(Abstraction("x", Variable("x")), Ket0())
        t = TensorPair(app, Ket1())
        assert t.is_value() == False


class TestLetTensorComplete:
    """Complete tests for LetTensor."""
    
    def test_str(self):
        lt = LetTensor("x", "y", TensorPair(Ket0(), Ket1()), Variable("x"))
        assert "let" in str(lt)
    
    def test_repr(self):
        lt = LetTensor("x", "y", TensorPair(Ket0(), Ket1()), Variable("x"))
        assert "LetTensor" in repr(lt)
    
    def test_substitution_shadowed(self):
        lt = LetTensor("x", "y", Variable("z"), Variable("x"))
        result = lt.substitute("x", Ket0())
        # x is bound, so body shouldn't change
        assert result.body == Variable("x")
    
    def test_substitution_in_tensor(self):
        lt = LetTensor("x", "y", Variable("z"), Variable("x"))
        result = lt.substitute("z", Ket0())
        assert result.tensor_term == Ket0()


class TestBangIntroComplete:
    """Complete tests for BangIntro."""
    
    def test_str(self):
        b = BangIntro(Ket0())
        assert "!" in str(b)
    
    def test_repr(self):
        b = BangIntro(Ket0())
        assert "BangIntro" in repr(b)
    
    def test_substitution(self):
        b = BangIntro(Variable("x"))
        result = b.substitute("x", Ket0())
        assert result.term == Ket0()
    
    def test_is_value_with_non_value(self):
        app = Application(Abstraction("x", Variable("x")), Ket0())
        b = BangIntro(app)
        assert b.is_value() == False


class TestLetBangComplete:
    """Complete tests for LetBang."""
    
    def test_str(self):
        lb = LetBang("x", BangIntro(Ket0()), Variable("x"))
        assert "let" in str(lb)
    
    def test_repr(self):
        lb = LetBang("x", BangIntro(Ket0()), Variable("x"))
        assert "LetBang" in repr(lb)
    
    def test_substitution_shadowed(self):
        lb = LetBang("x", Variable("y"), Variable("x"))
        result = lb.substitute("x", Ket0())
        # x is bound, body shouldn't change
        assert result.body == Variable("x")


class TestParagraphIntroComplete:
    """Complete tests for ParagraphIntro."""
    
    def test_str(self):
        p = ParagraphIntro(Ket0())
        assert "§" in str(p)
    
    def test_repr(self):
        p = ParagraphIntro(Ket0())
        assert "ParagraphIntro" in repr(p)
    
    def test_substitution(self):
        p = ParagraphIntro(Variable("x"))
        result = p.substitute("x", Ket0())
        assert result.term == Ket0()
    
    def test_is_value_with_non_value(self):
        app = Application(Abstraction("x", Variable("x")), Ket0())
        p = ParagraphIntro(app)
        assert p.is_value() == False


class TestUnitaryGateComplete:
    """Complete tests for UnitaryGate."""
    
    def test_y_matrix(self):
        m = UnitaryGate.Y.matrix()
        assert len(m) == 2
        assert m[0][1] == -1j
    
    def test_z_matrix(self):
        m = UnitaryGate.Z.matrix()
        assert m[0][0] == 1
        assert m[1][1] == -1
    
    def test_s_matrix(self):
        m = UnitaryGate.S.matrix()
        assert m[1][1] == 1j
    
    def test_t_matrix(self):
        m = UnitaryGate.T.matrix()
        assert len(m) == 2
    
    def test_cz_matrix(self):
        m = UnitaryGate.CZ.matrix()
        assert len(m) == 4
        assert m[3][3] == -1
    
    def test_swap_matrix(self):
        m = UnitaryGate.SWAP.matrix()
        assert len(m) == 4


class TestUnitaryAppComplete:
    """Complete tests for UnitaryApp."""
    
    def test_str(self):
        u = UnitaryApp(UnitaryGate.H, Ket0())
        assert "H" in str(u)
    
    def test_repr(self):
        u = UnitaryApp(UnitaryGate.H, Ket0())
        assert "UnitaryApp" in repr(u)


class TestQCtrlComplete:
    """Complete tests for QCtrl."""
    
    def test_str(self):
        q = QCtrl(Ket0(), Ket0(), Ket1())
        assert "qctrl" in str(q)
    
    def test_repr(self):
        q = QCtrl(Ket0(), Ket0(), Ket1())
        assert "QCtrl" in repr(q)


class TestMeasurementComplete:
    """Complete tests for Measurement."""
    
    def test_str(self):
        m = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        assert "meas" in str(m)
    
    def test_repr(self):
        m = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        assert "Measurement" in repr(m)
    
    def test_substitution(self):
        m = Measurement(MeasurementBasis.COMPUTATIONAL, Variable("x"))
        result = m.substitute("x", Ket0())
        assert result.arg == Ket0()


class TestNewComplete:
    """Complete tests for New."""
    
    def test_str(self):
        n = New()
        assert str(n) == "new"
    
    def test_repr(self):
        n = New()
        assert repr(n) == "New()"


class TestUnitComplete:
    """Complete tests for Unit."""
    
    def test_str(self):
        u = Unit()
        assert str(u) == "()"
    
    def test_repr(self):
        u = Unit()
        assert repr(u) == "Unit()"
    
    def test_substitute(self):
        u = Unit()
        result = u.substitute("x", Ket0())
        assert result == u


class TestInlComplete:
    """Complete tests for Inl."""
    
    def test_str(self):
        i = Inl(Ket0())
        assert "inl" in str(i)
    
    def test_repr(self):
        i = Inl(Ket0())
        assert "Inl" in repr(i)
    
    def test_substitution(self):
        i = Inl(Variable("x"))
        result = i.substitute("x", Ket0())
        assert result.term == Ket0()
    
    def test_is_value_with_non_value(self):
        app = Application(Abstraction("x", Variable("x")), Ket0())
        i = Inl(app)
        assert i.is_value() == False


class TestInrComplete:
    """Complete tests for Inr."""
    
    def test_str(self):
        i = Inr(Ket1())
        assert "inr" in str(i)
    
    def test_repr(self):
        i = Inr(Ket1())
        assert "Inr" in repr(i)
    
    def test_substitution(self):
        i = Inr(Variable("x"))
        result = i.substitute("x", Ket1())
        assert result.term == Ket1()
    
    def test_is_value_with_non_value(self):
        app = Application(Abstraction("x", Variable("x")), Ket0())
        i = Inr(app)
        assert i.is_value() == False


class TestCaseComplete:
    """Complete tests for Case."""
    
    def test_str(self):
        c = Case(Inl(Ket0()), "x", Variable("x"), "y", Variable("y"))
        assert "case" in str(c)
    
    def test_repr(self):
        c = Case(Inl(Ket0()), "x", Variable("x"), "y", Variable("y"))
        assert "Case" in repr(c)
    
    def test_substitution(self):
        c = Case(Variable("z"), "x", Variable("x"), "y", Variable("y"))
        result = c.substitute("z", Inl(Ket0()))
        assert result.scrutinee == Inl(Ket0())
    
    def test_substitution_shadowed_left(self):
        c = Case(Ket0(), "x", Variable("x"), "y", Variable("y"))
        result = c.substitute("x", Ket1())
        # x is bound in left branch
        assert result.branch_left == Variable("x")
    
    def test_substitution_shadowed_right(self):
        c = Case(Ket0(), "x", Variable("x"), "y", Variable("y"))
        result = c.substitute("y", Ket1())
        # y is bound in right branch
        assert result.branch_right == Variable("y")


class TestFreshVarGeneration:
    """Test fresh variable generation in Abstraction."""
    
    def test_fresh_var_with_collision(self):
        # Create a situation where x0 is already used
        body = Application(Variable("x0"), Variable("y"))
        lam = Abstraction("x", body)
        # Substitute y := x, requiring fresh var different from x and x0
        result = lam.substitute("y", Variable("x"))
        assert isinstance(result, Abstraction)
