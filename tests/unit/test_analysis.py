"""
Unit tests for λ^QLLR complexity and width analysis.
"""

import pytest
from qllr.core.syntax import (
    Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit
)
from qllr.core.types import (
    QubitType, UnitType, BoolType, LinearArrow, TensorProduct,
    SumType, BangType, ParagraphType, SharpType, ForallType
)
from qllr.analysis.complexity import (
    ComplexityBound, ComplexityAnalyzer, analyze_complexity, verify_polytime
)
from qllr.analysis.width import (
    WidthBound, WidthAnalyzer, analyze_width, compute_qubit_count
)
import cmath


class TestComplexityBound:
    """Tests for complexity bounds."""
    
    def test_bound_creation(self):
        bound = ComplexityBound(exponent=2)
        assert bound.exponent == 2
        assert bound.coefficient == 1
    
    def test_bound_with_coefficient(self):
        bound = ComplexityBound(exponent=3, coefficient=5)
        assert bound.coefficient == 5
    
    def test_evaluate(self):
        bound = ComplexityBound(exponent=2, coefficient=1)
        assert bound.evaluate(3) == 9  # 3^2
        
        bound2 = ComplexityBound(exponent=2, coefficient=2)
        assert bound2.evaluate(3) == 18  # 2 * 3^2
    
    def test_compose(self):
        b1 = ComplexityBound(exponent=2)
        b2 = ComplexityBound(exponent=3)
        composed = b1.compose(b2)
        assert composed.exponent == 4  # max(2,3) + 1
    
    def test_parallel(self):
        b1 = ComplexityBound(exponent=2)
        b2 = ComplexityBound(exponent=3)
        parallel = b1.parallel(b2)
        assert parallel.exponent == 3  # max(2,3)
    
    def test_str(self):
        bound = ComplexityBound(exponent=2)
        assert "O(n^2)" in str(bound)
        
        bound2 = ComplexityBound(exponent=3, coefficient=5)
        assert "5" in str(bound2)
    
    def test_repr(self):
        bound = ComplexityBound(exponent=2)
        assert "ComplexityBound" in repr(bound)


class TestComplexityAnalyzer:
    """Tests for complexity analysis."""
    
    def test_analyze_qubit_type(self):
        analyzer = ComplexityAnalyzer()
        bound = analyzer.analyze_type(QubitType())
        # Modal depth 0, exponent = 2^(0+1) = 2
        assert bound.exponent == 2
    
    def test_analyze_paragraph_type(self):
        analyzer = ComplexityAnalyzer()
        bound = analyzer.analyze_type(ParagraphType(QubitType()))
        # Modal depth 1, exponent = 2^(1+1) = 4
        assert bound.exponent == 4
    
    def test_analyze_nested_paragraph(self):
        analyzer = ComplexityAnalyzer()
        typ = ParagraphType(ParagraphType(QubitType()))
        bound = analyzer.analyze_type(typ)
        # Modal depth 2, exponent = 2^(2+1) = 8
        assert bound.exponent == 8
    
    def test_analyze_linear_arrow(self):
        analyzer = ComplexityAnalyzer()
        typ = LinearArrow(QubitType(), QubitType())
        bound = analyzer.analyze_type(typ)
        assert bound.exponent == 2  # max of domain and codomain depths
    
    def test_analyze_term_ket0(self):
        analyzer = ComplexityAnalyzer()
        bound = analyzer.analyze_term(Ket0())
        assert bound.exponent >= 2
    
    def test_analyze_term_abstraction(self):
        analyzer = ComplexityAnalyzer()
        term = Abstraction("x", Variable("x"))
        bound = analyzer.analyze_term(term)
        assert bound.exponent >= 2
    
    def test_analyze_term_paragraph(self):
        analyzer = ComplexityAnalyzer()
        term = ParagraphIntro(Ket0())
        bound = analyzer.analyze_term(term)
        # Paragraph increases depth
        assert bound.exponent >= 4
    
    def test_term_size_ket(self):
        analyzer = ComplexityAnalyzer()
        assert analyzer._term_size(Ket0()) == 1
        assert analyzer._term_size(Ket1()) == 1
    
    def test_term_size_variable(self):
        analyzer = ComplexityAnalyzer()
        assert analyzer._term_size(Variable("x")) == 1
    
    def test_term_size_abstraction(self):
        analyzer = ComplexityAnalyzer()
        term = Abstraction("x", Variable("x"))
        assert analyzer._term_size(term) == 2
    
    def test_term_size_application(self):
        analyzer = ComplexityAnalyzer()
        term = Application(Variable("f"), Variable("x"))
        assert analyzer._term_size(term) == 3
    
    def test_term_size_tensor(self):
        analyzer = ComplexityAnalyzer()
        term = TensorPair(Ket0(), Ket1())
        assert analyzer._term_size(term) == 3
    
    def test_term_size_qctrl(self):
        analyzer = ComplexityAnalyzer()
        term = QCtrl(Ket0(), Ket0(), Ket1())
        assert analyzer._term_size(term) == 4
    
    def test_term_depth_ket(self):
        analyzer = ComplexityAnalyzer()
        assert analyzer._term_depth(Ket0()) == 0
    
    def test_term_depth_paragraph(self):
        analyzer = ComplexityAnalyzer()
        term = ParagraphIntro(Ket0())
        assert analyzer._term_depth(term) == 1
    
    def test_term_depth_nested_paragraph(self):
        analyzer = ComplexityAnalyzer()
        term = ParagraphIntro(ParagraphIntro(Ket0()))
        assert analyzer._term_depth(term) == 2
    
    def test_term_depth_bang(self):
        analyzer = ComplexityAnalyzer()
        term = BangIntro(Ket0())
        assert analyzer._term_depth(term) == 0  # Bang doesn't increase depth
    
    def test_verify_polytime_valid(self):
        analyzer = ComplexityAnalyzer()
        term = Ket0()
        typ = QubitType()
        is_poly, bound = analyzer.verify_polytime(term, typ)
        assert is_poly == True
    
    def test_estimate_reduction_steps(self):
        analyzer = ComplexityAnalyzer()
        term = Abstraction("x", Variable("x"))
        steps = analyzer.estimate_reduction_steps(term)
        assert steps > 0
    
    def test_is_in_polytime_fragment_ket(self):
        analyzer = ComplexityAnalyzer()
        assert analyzer.is_in_polytime_fragment(Ket0()) == True
    
    def test_is_in_polytime_fragment_paragraph(self):
        analyzer = ComplexityAnalyzer()
        term = ParagraphIntro(Ket0())
        assert analyzer.is_in_polytime_fragment(term) == True
    
    def test_is_in_polytime_fragment_bang_at_depth0(self):
        analyzer = ComplexityAnalyzer()
        term = BangIntro(Ket0())
        assert analyzer.is_in_polytime_fragment(term) == True
    
    def test_is_in_polytime_fragment_bang_nested(self):
        analyzer = ComplexityAnalyzer()
        # Bang inside paragraph (depth > 0)
        term = ParagraphIntro(BangIntro(Ket0()))
        assert analyzer.is_in_polytime_fragment(term) == False
    
    def test_analyze_complexity_convenience(self):
        bound = analyze_complexity(Ket0())
        assert bound.exponent >= 2
    
    def test_verify_polytime_convenience(self):
        is_poly = verify_polytime(Ket0(), QubitType())
        assert is_poly == True


class TestWidthBound:
    """Tests for width bounds."""
    
    def test_width_creation(self):
        bound = WidthBound(3)
        assert bound.width == 3
        assert bound.peak_width == 3
    
    def test_width_with_peak(self):
        bound = WidthBound(3, peak_width=5)
        assert bound.width == 3
        assert bound.peak_width == 5
    
    def test_sequential(self):
        b1 = WidthBound(2)
        b2 = WidthBound(3)
        seq = b1.sequential(b2)
        assert seq.width == 3  # max
    
    def test_parallel(self):
        b1 = WidthBound(2)
        b2 = WidthBound(3)
        par = b1.parallel(b2)
        assert par.width == 5  # sum
    
    def test_add_control(self):
        bound = WidthBound(2)
        with_ctrl = bound.add_control()
        assert with_ctrl.width == 3  # +1 for control
    
    def test_str(self):
        bound = WidthBound(3)
        assert "width=3" in str(bound)
    
    def test_str_with_peak(self):
        bound = WidthBound(3, peak_width=5)
        assert "peak=5" in str(bound)
    
    def test_repr(self):
        bound = WidthBound(3)
        assert "WidthBound" in repr(bound)


class TestWidthAnalyzer:
    """Tests for width (space) analysis."""
    
    def test_analyze_qubit_type(self):
        analyzer = WidthAnalyzer()
        bound = analyzer.analyze_type(QubitType())
        assert bound.width == 1
    
    def test_analyze_unit_type(self):
        analyzer = WidthAnalyzer()
        bound = analyzer.analyze_type(UnitType())
        assert bound.width == 0
    
    def test_analyze_tensor_type(self):
        analyzer = WidthAnalyzer()
        typ = TensorProduct(QubitType(), QubitType())
        bound = analyzer.analyze_type(typ)
        assert bound.width == 2  # sum
    
    def test_analyze_linear_arrow_type(self):
        analyzer = WidthAnalyzer()
        typ = LinearArrow(QubitType(), QubitType())
        bound = analyzer.analyze_type(typ)
        assert bound.width == 1  # max
    
    def test_analyze_sum_type(self):
        analyzer = WidthAnalyzer()
        typ = SumType(QubitType(), QubitType())
        bound = analyzer.analyze_type(typ)
        assert bound.width == 1  # max (only one active)
    
    def test_analyze_bang_type(self):
        analyzer = WidthAnalyzer()
        typ = BangType(QubitType())
        bound = analyzer.analyze_type(typ)
        assert bound.width == 1
    
    def test_analyze_forall_type(self):
        analyzer = WidthAnalyzer()
        typ = ForallType("X", QubitType())
        bound = analyzer.analyze_type(typ)
        assert bound.width == 1
    
    def test_analyze_term_ket0(self):
        analyzer = WidthAnalyzer()
        bound = analyzer.analyze_term(Ket0())
        assert bound.width == 1
    
    def test_analyze_term_new(self):
        analyzer = WidthAnalyzer()
        bound = analyzer.analyze_term(New())
        assert bound.width == 1
    
    def test_analyze_term_unit(self):
        analyzer = WidthAnalyzer()
        bound = analyzer.analyze_term(Unit())
        assert bound.width == 0
    
    def test_analyze_term_variable(self):
        analyzer = WidthAnalyzer()
        bound = analyzer.analyze_term(Variable("x"), {"x": 2})
        assert bound.width == 2
    
    def test_analyze_term_tensor(self):
        analyzer = WidthAnalyzer()
        term = TensorPair(Ket0(), Ket1())
        bound = analyzer.analyze_term(term)
        assert bound.width == 2  # parallel
    
    def test_analyze_term_superposition(self):
        analyzer = WidthAnalyzer()
        sqrt2 = 1/cmath.sqrt(2)
        term = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
        bound = analyzer.analyze_term(term)
        assert bound.width >= 1
    
    def test_analyze_term_abstraction(self):
        analyzer = WidthAnalyzer()
        term = Abstraction("x", Variable("x"))
        bound = analyzer.analyze_term(term)
        assert bound.width >= 1
    
    def test_analyze_term_application(self):
        analyzer = WidthAnalyzer()
        term = Application(Variable("f"), Ket0())
        bound = analyzer.analyze_term(term)
        assert bound.width >= 1
    
    def test_analyze_term_qctrl(self):
        analyzer = WidthAnalyzer()
        term = QCtrl(Ket0(), Ket0(), Ket1())
        bound = analyzer.analyze_term(term)
        # Control + branches + extra for control qubit
        assert bound.width >= 2
    
    def test_analyze_term_unitary(self):
        analyzer = WidthAnalyzer()
        term = UnitaryApp(UnitaryGate.H, Ket0())
        bound = analyzer.analyze_term(term)
        assert bound.width == 1  # Unitary preserves width
    
    def test_analyze_term_let_tensor(self):
        analyzer = WidthAnalyzer()
        tensor = TensorPair(Ket0(), Ket1())
        term = LetTensor("x", "y", tensor, Variable("x"))
        bound = analyzer.analyze_term(term)
        assert bound.width >= 2
    
    def test_analyze_term_let_bang(self):
        analyzer = WidthAnalyzer()
        term = LetBang("x", BangIntro(Ket0()), Variable("x"))
        bound = analyzer.analyze_term(term)
        assert bound.width >= 1
    
    def test_analyze_term_bang_intro(self):
        analyzer = WidthAnalyzer()
        term = BangIntro(Ket0())
        bound = analyzer.analyze_term(term)
        assert bound.width == 1
    
    def test_analyze_term_paragraph_intro(self):
        analyzer = WidthAnalyzer()
        term = ParagraphIntro(Ket0())
        bound = analyzer.analyze_term(term)
        assert bound.width == 1
    
    def test_analyze_term_measurement(self):
        analyzer = WidthAnalyzer()
        term = Measurement(MeasurementBasis.COMPUTATIONAL, Ket0())
        bound = analyzer.analyze_term(term)
        assert bound.width == 1
    
    def test_verify_width_bound_pass(self):
        analyzer = WidthAnalyzer()
        satisfies, actual = analyzer.verify_width_bound(Ket0(), 5)
        assert satisfies == True
        assert actual.width == 1
    
    def test_verify_width_bound_fail(self):
        analyzer = WidthAnalyzer()
        term = TensorPair(TensorPair(Ket0(), Ket1()), Ket0())  # 3 qubits
        satisfies, actual = analyzer.verify_width_bound(term, 2)
        assert satisfies == False
    
    def test_compute_circuit_width(self):
        analyzer = WidthAnalyzer()
        term = TensorPair(Ket0(), Ket1())
        width = analyzer.compute_circuit_width(term)
        assert width == 2
    
    def test_estimate_ancilla_count_basic(self):
        analyzer = WidthAnalyzer()
        assert analyzer.estimate_ancilla_count(Ket0()) == 0
    
    def test_estimate_ancilla_count_qctrl(self):
        analyzer = WidthAnalyzer()
        term = QCtrl(Ket0(), Ket0(), Ket1())
        ancilla = analyzer.estimate_ancilla_count(term)
        assert ancilla >= 1  # At least 1 for the control
    
    def test_estimate_ancilla_count_nested_qctrl(self):
        analyzer = WidthAnalyzer()
        inner = QCtrl(Ket0(), Ket0(), Ket1())
        outer = QCtrl(Ket0(), inner, Ket0())
        ancilla = analyzer.estimate_ancilla_count(outer)
        assert ancilla >= 2  # At least 2 for nested controls
    
    def test_analyze_width_convenience(self):
        bound = analyze_width(Ket0())
        assert bound.width == 1
    
    def test_compute_qubit_count_convenience(self):
        count = compute_qubit_count(TensorPair(Ket0(), Ket1()))
        assert count == 2


class TestComplexTermAnalysis:
    """Tests for analyzing complex terms."""
    
    def test_bell_state_width(self):
        analyzer = WidthAnalyzer()
        # H[|0⟩] ⊗ |0⟩
        term = TensorPair(UnitaryApp(UnitaryGate.H, Ket0()), Ket0())
        bound = analyzer.analyze_term(term)
        assert bound.width == 2
    
    def test_nested_tensor_width(self):
        analyzer = WidthAnalyzer()
        # ((|0⟩ ⊗ |1⟩) ⊗ |0⟩)
        term = TensorPair(TensorPair(Ket0(), Ket1()), Ket0())
        bound = analyzer.analyze_term(term)
        assert bound.width == 3
    
    def test_abstraction_with_tensor(self):
        width_analyzer = WidthAnalyzer()
        # λx. x ⊗ |0⟩
        term = Abstraction("x", TensorPair(Variable("x"), Ket0()))
        bound = width_analyzer.analyze_term(term)
        assert bound.width == 2
    
    def test_chained_unitaries_complexity(self):
        comp_analyzer = ComplexityAnalyzer()
        # H[X[Y[Z[|0⟩]]]]
        term = UnitaryApp(UnitaryGate.H,
                         UnitaryApp(UnitaryGate.X,
                                   UnitaryApp(UnitaryGate.Y,
                                             UnitaryApp(UnitaryGate.Z, Ket0()))))
        bound = comp_analyzer.analyze_term(term)
        assert bound.exponent >= 2
