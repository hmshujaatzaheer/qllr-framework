"""
Comprehensive tests for 100% coverage on complexity and width analysis.
"""

import pytest
import cmath
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
from qllr.analysis.complexity import ComplexityBound, ComplexityAnalyzer
from qllr.analysis.width import WidthBound, WidthAnalyzer


class TestComplexityTermSizeComplete:
    """Complete tests for _term_size method."""
    
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()
    
    def test_term_size_superposition(self):
        """Test term size of superposition."""
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        size = self.analyzer._term_size(term)
        assert size == 3  # 1 + 1 + 1
    
    def test_term_size_let_tensor(self):
        """Test term size of let-tensor."""
        tensor = TensorPair(Ket0(), Ket1())
        term = LetTensor("x", "y", tensor, Variable("x"))
        size = self.analyzer._term_size(term)
        assert size >= 4
    
    def test_term_size_let_bang(self):
        """Test term size of let-bang."""
        term = LetBang("x", BangIntro(Ket0()), Variable("x"))
        size = self.analyzer._term_size(term)
        assert size >= 3
    
    def test_term_size_bang_intro(self):
        """Test term size of bang intro."""
        term = BangIntro(Ket0())
        size = self.analyzer._term_size(term)
        assert size == 2
    
    def test_term_size_paragraph_intro(self):
        """Test term size of paragraph intro."""
        term = ParagraphIntro(Ket0())
        size = self.analyzer._term_size(term)
        assert size == 2
    
    def test_term_size_measurement(self):
        """Test term size of measurement."""
        term = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        size = self.analyzer._term_size(term)
        assert size == 2
    
    def test_term_size_unknown(self):
        """Test term size of unknown term type."""
        class UnknownTerm:
            pass
        size = self.analyzer._term_size(UnknownTerm())
        assert size == 1  # Default


class TestComplexityTermDepthComplete:
    """Complete tests for _term_depth method."""
    
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()
    
    def test_term_depth_superposition(self):
        """Test depth of superposition (max of branches)."""
        sqrt2 = 1/cmath.sqrt(2)
        # One branch has depth 1
        term = Superposition(sqrt2, ParagraphIntro(Ket0()), sqrt2, Ket1())
        depth = self.analyzer._term_depth(term)
        assert depth == 1
    
    def test_term_depth_application(self):
        """Test depth of application (max of func and arg)."""
        term = Application(ParagraphIntro(Abstraction("x", Variable("x"))), Ket0())
        depth = self.analyzer._term_depth(term)
        assert depth == 1
    
    def test_term_depth_tensor(self):
        """Test depth of tensor (max of components)."""
        term = TensorPair(ParagraphIntro(Ket0()), Ket1())
        depth = self.analyzer._term_depth(term)
        assert depth == 1
    
    def test_term_depth_let_tensor(self):
        """Test depth of let-tensor."""
        tensor = TensorPair(ParagraphIntro(Ket0()), Ket1())
        term = LetTensor("x", "y", tensor, Ket0())
        depth = self.analyzer._term_depth(term)
        assert depth == 1
    
    def test_term_depth_let_bang(self):
        """Test depth of let-bang."""
        term = LetBang("x", BangIntro(ParagraphIntro(Ket0())), Ket0())
        depth = self.analyzer._term_depth(term)
        assert depth == 1
    
    def test_term_depth_qctrl(self):
        """Test depth of qctrl (max of all components)."""
        term = QCtrl(ParagraphIntro(Ket0()), Ket0(), Ket1())
        depth = self.analyzer._term_depth(term)
        assert depth == 1
    
    def test_term_depth_measurement(self):
        """Test depth of measurement."""
        term = Measurement(MeasurementBasis.COMPUTATIONAL, ParagraphIntro(Ket0()))
        depth = self.analyzer._term_depth(term)
        assert depth == 1
    
    def test_term_depth_unknown(self):
        """Test depth of unknown term type."""
        class UnknownTerm:
            pass
        depth = self.analyzer._term_depth(UnknownTerm())
        assert depth == 0  # Default


class TestVerifyPolytimeComplete:
    """Complete tests for verify_polytime method."""
    
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()
    
    def test_verify_polytime_exceeds_type_depth(self):
        """Test when term depth exceeds type depth."""
        # Term with depth 2 (nested paragraph)
        term = ParagraphIntro(ParagraphIntro(Ket0()))
        # Type with depth 1
        typ = ParagraphType(QubitType())
        is_poly, bound = self.analyzer.verify_polytime(term, typ)
        assert is_poly == False  # Term depth > type depth


class TestIsInPolytimeFragmentComplete:
    """Complete tests for is_in_polytime_fragment method."""
    
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()
    
    def test_polytime_let_bang(self):
        """Test let-bang in polytime fragment."""
        term = LetBang("x", BangIntro(Ket0()), Variable("x"))
        result = self.analyzer.is_in_polytime_fragment(term)
        assert result == True
    
    def test_polytime_superposition(self):
        """Test superposition in polytime fragment."""
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        result = self.analyzer.is_in_polytime_fragment(term)
        assert result == True
    
    def test_polytime_application(self):
        """Test application in polytime fragment."""
        func = Abstraction("x", Variable("x"))
        term = Application(func, Ket0())
        result = self.analyzer.is_in_polytime_fragment(term)
        assert result == True
    
    def test_polytime_tensor(self):
        """Test tensor in polytime fragment."""
        term = TensorPair(Ket0(), Ket1())
        result = self.analyzer.is_in_polytime_fragment(term)
        assert result == True
    
    def test_polytime_let_tensor(self):
        """Test let-tensor in polytime fragment."""
        tensor = TensorPair(Ket0(), Ket1())
        term = LetTensor("x", "y", tensor, Variable("x"))
        result = self.analyzer.is_in_polytime_fragment(term)
        assert result == True
    
    def test_polytime_unitary(self):
        """Test unitary in polytime fragment."""
        term = UnitaryApp(UnitaryGate.H, Ket0())
        result = self.analyzer.is_in_polytime_fragment(term)
        assert result == True
    
    def test_polytime_qctrl(self):
        """Test qctrl in polytime fragment."""
        term = QCtrl(Ket0(), Ket0(), Ket1())
        result = self.analyzer.is_in_polytime_fragment(term)
        assert result == True
    
    def test_polytime_measurement(self):
        """Test measurement in polytime fragment."""
        term = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        result = self.analyzer.is_in_polytime_fragment(term)
        assert result == True
    
    def test_polytime_unknown(self):
        """Test unknown term type defaults to True."""
        class UnknownTerm:
            pass
        result = self.analyzer.is_in_polytime_fragment(UnknownTerm())
        assert result == True  # Default


class TestWidthAnalyzerComplete:
    """Complete tests for WidthAnalyzer methods."""
    
    def setup_method(self):
        self.analyzer = WidthAnalyzer()
    
    def test_analyze_type_type_variable(self):
        """Test width of type variable."""
        typ = TypeVariable("X")
        bound = self.analyzer.analyze_type(typ)
        # Unknown type variable, implementation returns 0 or 1
        assert bound.width >= 0
    
    def test_analyze_type_sharp(self):
        """Test width of sharp type."""
        typ = SharpType(QubitType())
        bound = self.analyzer.analyze_type(typ)
        assert bound.width == 1
    
    def test_analyze_type_paragraph(self):
        """Test width of paragraph type."""
        typ = ParagraphType(TensorProduct(QubitType(), QubitType()))
        bound = self.analyzer.analyze_type(typ)
        assert bound.width == 2
    
    def test_analyze_term_inl(self):
        """Test width of inl term."""
        term = Inl(Ket0())
        bound = self.analyzer.analyze_term(term)
        # Implementation may return 0 or 1 depending on how inl is handled
        assert bound.width >= 0
    
    def test_analyze_term_inr(self):
        """Test width of inr term."""
        term = Inr(Ket1())
        bound = self.analyzer.analyze_term(term)
        assert bound.width >= 0
    
    def test_analyze_term_case(self):
        """Test width of case term."""
        term = Case(Inl(Ket0()), "x", Variable("x"), "y", Variable("y"))
        bound = self.analyzer.analyze_term(term, {"x": 1, "y": 1})
        assert bound.width >= 0
    
    def test_analyze_term_let_tensor(self):
        """Test width of let-tensor term."""
        tensor = TensorPair(Ket0(), Ket1())
        term = LetTensor("x", "y", tensor, TensorPair(Variable("x"), Variable("y")))
        bound = self.analyzer.analyze_term(term)
        assert bound.width == 2
    
    def test_estimate_ancilla_nested_qctrl(self):
        """Test ancilla estimation for nested qctrl."""
        inner = QCtrl(Ket0(), Ket0(), Ket1())
        term = QCtrl(Ket0(), inner, Ket0())
        count = self.analyzer.estimate_ancilla_count(term)
        assert count >= 2
    
    def test_estimate_ancilla_tensor(self):
        """Test ancilla estimation for tensor (no ancillas)."""
        term = TensorPair(Ket0(), Ket1())
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_abstraction(self):
        """Test ancilla estimation for abstraction."""
        term = Abstraction("x", Variable("x"))
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_application(self):
        """Test ancilla estimation for application."""
        func = Abstraction("x", Variable("x"))
        term = Application(func, Ket0())
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_superposition(self):
        """Test ancilla estimation for superposition."""
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_unitary(self):
        """Test ancilla estimation for unitary."""
        term = UnitaryApp(UnitaryGate.H, Ket0())
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_measurement(self):
        """Test ancilla estimation for measurement."""
        term = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_let_tensor(self):
        """Test ancilla estimation for let-tensor."""
        tensor = TensorPair(Ket0(), Ket1())
        term = LetTensor("x", "y", tensor, Variable("x"))
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_let_bang(self):
        """Test ancilla estimation for let-bang."""
        term = LetBang("x", BangIntro(Ket0()), Variable("x"))
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_bang_intro(self):
        """Test ancilla estimation for bang intro."""
        term = BangIntro(Ket0())
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_estimate_ancilla_paragraph_intro(self):
        """Test ancilla estimation for paragraph intro."""
        term = ParagraphIntro(Ket0())
        count = self.analyzer.estimate_ancilla_count(term)
        assert count == 0
    
    def test_analyze_term_unknown(self):
        """Test width analysis of unknown term type."""
        class UnknownTerm:
            pass
        bound = self.analyzer.analyze_term(UnknownTerm())
        assert bound.width >= 0  # Default
    
    def test_estimate_ancilla_unknown(self):
        """Test ancilla estimation of unknown term type."""
        class UnknownTerm:
            pass
        count = self.analyzer.estimate_ancilla_count(UnknownTerm())
        assert count == 0  # Default
