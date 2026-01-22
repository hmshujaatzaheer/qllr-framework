"""
Additional tests for 100% coverage on types module.
"""

import pytest
from qllr.core.types import (
    Type, QubitType, UnitType, BoolType, TypeVariable,
    LinearArrow, TensorProduct, SumType, BangType, ParagraphType,
    SharpType, ForallType, types_equal
)


class TestQubitTypeComplete:
    """Complete tests for QubitType."""
    
    def test_width(self):
        q = QubitType()
        assert q.width() == 1
    
    def test_repr(self):
        q = QubitType()
        assert repr(q) == "QubitType()"
    
    def test_str(self):
        q = QubitType()
        assert str(q) == "qubit"


class TestUnitTypeComplete:
    """Complete tests for UnitType."""
    
    def test_free_type_variables(self):
        u = UnitType()
        assert u.free_type_variables() == set()
    
    def test_substitute_type(self):
        u = UnitType()
        result = u.substitute_type("X", QubitType())
        assert result == u
    
    def test_modal_depth(self):
        u = UnitType()
        assert u.modal_depth() == 0
    
    def test_is_linear(self):
        u = UnitType()
        assert u.is_linear() == False
    
    def test_str(self):
        u = UnitType()
        assert str(u) == "1"
    
    def test_repr(self):
        u = UnitType()
        assert repr(u) == "UnitType()"


class TestBoolTypeComplete:
    """Complete tests for BoolType."""
    
    def test_free_type_variables(self):
        b = BoolType()
        assert b.free_type_variables() == set()
    
    def test_substitute_type(self):
        b = BoolType()
        result = b.substitute_type("X", QubitType())
        assert result == b
    
    def test_modal_depth(self):
        b = BoolType()
        assert b.modal_depth() == 0
    
    def test_is_linear(self):
        b = BoolType()
        assert b.is_linear() == False
    
    def test_str(self):
        b = BoolType()
        assert str(b) == "bool"
    
    def test_repr(self):
        b = BoolType()
        assert repr(b) == "BoolType()"


class TestTypeVariableComplete:
    """Complete tests for TypeVariable."""
    
    def test_free_type_variables(self):
        tv = TypeVariable("X")
        assert tv.free_type_variables() == {"X"}
    
    def test_modal_depth(self):
        tv = TypeVariable("X")
        assert tv.modal_depth() == 0
    
    def test_is_linear(self):
        tv = TypeVariable("X")
        assert tv.is_linear() == True
    
    def test_str(self):
        tv = TypeVariable("X")
        assert str(tv) == "X"
    
    def test_repr(self):
        tv = TypeVariable("X")
        assert repr(tv) == "TypeVariable('X')"


class TestLinearArrowComplete:
    """Complete tests for LinearArrow."""
    
    def test_free_type_variables(self):
        arr = LinearArrow(TypeVariable("X"), TypeVariable("Y"))
        assert arr.free_type_variables() == {"X", "Y"}
    
    def test_width(self):
        arr = LinearArrow(QubitType(), QubitType())
        assert arr.width() == 1
    
    def test_width_with_tensor(self):
        arr = LinearArrow(TensorProduct(QubitType(), QubitType()), QubitType())
        assert arr.width() == 2
    
    def test_str(self):
        arr = LinearArrow(QubitType(), QubitType())
        assert "⊸" in str(arr)
    
    def test_repr(self):
        arr = LinearArrow(QubitType(), QubitType())
        assert "LinearArrow" in repr(arr)


class TestTensorProductComplete:
    """Complete tests for TensorProduct."""
    
    def test_free_type_variables(self):
        t = TensorProduct(TypeVariable("X"), TypeVariable("Y"))
        assert t.free_type_variables() == {"X", "Y"}
    
    def test_substitute_type(self):
        t = TensorProduct(TypeVariable("X"), TypeVariable("X"))
        result = t.substitute_type("X", QubitType())
        assert isinstance(result, TensorProduct)
        assert result.left == QubitType()
    
    def test_width(self):
        t = TensorProduct(QubitType(), QubitType())
        assert t.width() == 2
    
    def test_str(self):
        t = TensorProduct(QubitType(), QubitType())
        assert "⊗" in str(t)
    
    def test_repr(self):
        t = TensorProduct(QubitType(), QubitType())
        assert "TensorProduct" in repr(t)


class TestSumTypeComplete:
    """Complete tests for SumType."""
    
    def test_free_type_variables(self):
        s = SumType(TypeVariable("X"), TypeVariable("Y"))
        assert s.free_type_variables() == {"X", "Y"}
    
    def test_substitute_type(self):
        s = SumType(TypeVariable("X"), QubitType())
        result = s.substitute_type("X", BoolType())
        assert isinstance(result, SumType)
        assert result.left == BoolType()
    
    def test_modal_depth(self):
        s = SumType(ParagraphType(QubitType()), QubitType())
        assert s.modal_depth() == 1
    
    def test_is_linear(self):
        s = SumType(QubitType(), UnitType())
        assert s.is_linear() == True
    
    def test_width(self):
        s = SumType(QubitType(), TensorProduct(QubitType(), QubitType()))
        assert s.width() == 2
    
    def test_str(self):
        s = SumType(QubitType(), QubitType())
        assert "⊕" in str(s)
    
    def test_repr(self):
        s = SumType(QubitType(), QubitType())
        assert "SumType" in repr(s)


class TestBangTypeComplete:
    """Complete tests for BangType."""
    
    def test_free_type_variables(self):
        b = BangType(TypeVariable("X"))
        assert b.free_type_variables() == {"X"}
    
    def test_substitute_type(self):
        b = BangType(TypeVariable("X"))
        result = b.substitute_type("X", QubitType())
        assert isinstance(result, BangType)
        assert result.inner == QubitType()
    
    def test_width(self):
        b = BangType(QubitType())
        assert b.width() == 1
    
    def test_str(self):
        b = BangType(QubitType())
        assert str(b) == "!qubit"
    
    def test_repr(self):
        b = BangType(QubitType())
        assert "BangType" in repr(b)


class TestParagraphTypeComplete:
    """Complete tests for ParagraphType."""
    
    def test_free_type_variables(self):
        p = ParagraphType(TypeVariable("X"))
        assert p.free_type_variables() == {"X"}
    
    def test_substitute_type(self):
        p = ParagraphType(TypeVariable("X"))
        result = p.substitute_type("X", QubitType())
        assert isinstance(result, ParagraphType)
        assert result.inner == QubitType()
    
    def test_width(self):
        p = ParagraphType(QubitType())
        assert p.width() == 1
    
    def test_str(self):
        p = ParagraphType(QubitType())
        assert str(p) == "§qubit"
    
    def test_repr(self):
        p = ParagraphType(QubitType())
        assert "ParagraphType" in repr(p)


class TestSharpTypeComplete:
    """Complete tests for SharpType."""
    
    def test_free_type_variables(self):
        s = SharpType(TypeVariable("X"))
        assert s.free_type_variables() == {"X"}
    
    def test_substitute_type(self):
        s = SharpType(TypeVariable("X"))
        result = s.substitute_type("X", QubitType())
        assert isinstance(result, SharpType)
        assert result.inner == QubitType()
    
    def test_modal_depth(self):
        s = SharpType(ParagraphType(QubitType()))
        assert s.modal_depth() == 1
    
    def test_width(self):
        s = SharpType(QubitType())
        assert s.width() == 1
    
    def test_str(self):
        s = SharpType(QubitType())
        assert str(s) == "♯qubit"
    
    def test_repr(self):
        s = SharpType(QubitType())
        assert "SharpType" in repr(s)


class TestForallTypeComplete:
    """Complete tests for ForallType."""
    
    def test_substitute_type_shadowed(self):
        f = ForallType("X", TypeVariable("X"))
        result = f.substitute_type("X", QubitType())
        # X is bound, so no substitution
        assert result == f
    
    def test_substitute_type_with_capture(self):
        # ∀X. (X → Y) with Y := X needs alpha-conversion
        f = ForallType("X", LinearArrow(TypeVariable("X"), TypeVariable("Y")))
        result = f.substitute_type("Y", TypeVariable("X"))
        # Should alpha-convert to avoid capture
        assert isinstance(result, ForallType)
    
    def test_modal_depth(self):
        f = ForallType("X", ParagraphType(TypeVariable("X")))
        assert f.modal_depth() == 1
    
    def test_is_linear(self):
        f = ForallType("X", QubitType())
        assert f.is_linear() == True
    
    def test_str(self):
        f = ForallType("X", TypeVariable("X"))
        assert "∀" in str(f)
    
    def test_repr(self):
        f = ForallType("X", TypeVariable("X"))
        assert "ForallType" in repr(f)


class TestTypesEqualComplete:
    """Complete tests for types_equal function."""
    
    def test_equal_unit_types(self):
        assert types_equal(UnitType(), UnitType())
    
    def test_equal_bool_types(self):
        assert types_equal(BoolType(), BoolType())
    
    def test_equal_type_variables(self):
        assert types_equal(TypeVariable("X"), TypeVariable("X"))
        assert not types_equal(TypeVariable("X"), TypeVariable("Y"))
    
    def test_equal_sum_types(self):
        s1 = SumType(QubitType(), QubitType())
        s2 = SumType(QubitType(), QubitType())
        assert types_equal(s1, s2)
    
    def test_equal_bang_types(self):
        b1 = BangType(QubitType())
        b2 = BangType(QubitType())
        assert types_equal(b1, b2)
    
    def test_equal_paragraph_types(self):
        p1 = ParagraphType(QubitType())
        p2 = ParagraphType(QubitType())
        assert types_equal(p1, p2)
    
    def test_equal_sharp_types(self):
        s1 = SharpType(QubitType())
        s2 = SharpType(QubitType())
        assert types_equal(s1, s2)
    
    def test_equal_forall_alpha(self):
        # Alpha-equivalent forall types
        f1 = ForallType("X", TypeVariable("X"))
        f2 = ForallType("Y", TypeVariable("Y"))
        assert types_equal(f1, f2)
    
    def test_not_equal_different_types(self):
        assert not types_equal(QubitType(), BoolType())
        assert not types_equal(BangType(QubitType()), ParagraphType(QubitType()))
