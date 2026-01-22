"""
Example Quantum Programs in λ^QLLR

This module demonstrates various quantum programs expressible in the QLLR framework.
"""

from qllr.core import (
    Ket0, Ket1, Variable, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement, New
)
from qllr.typing import TypeChecker, OrthogonalityChecker
from qllr.compilation import extract_circuit
from qllr.analysis import ComplexityAnalyzer, WidthAnalyzer
import cmath


def example_hadamard():
    """
    Example 1: Hadamard gate on |0⟩
    
    Creates the superposition |+⟩ = (|0⟩ + |1⟩)/√2
    """
    print("=" * 50)
    print("Example 1: Hadamard Gate")
    print("=" * 50)
    
    term = UnitaryApp(UnitaryGate.H, Ket0())
    print(f"Term: H[|0⟩]")
    
    # Type check
    tc = TypeChecker()
    typ = tc.check(term)
    print(f"Type: {typ}")
    
    # Extract circuit
    circuit = extract_circuit(term)
    print(f"Circuit: {circuit.size()} gates, {circuit.num_qubits} qubits")
    print(f"OpenQASM:\n{circuit.to_openqasm()}")
    
    return term


def example_bell_state():
    """
    Example 2: Bell State Preparation (partial)
    
    Creates H|0⟩ ⊗ |0⟩, the first step of Bell state preparation.
    """
    print("\n" + "=" * 50)
    print("Example 2: Bell State (Initial)")
    print("=" * 50)
    
    h_qubit = UnitaryApp(UnitaryGate.H, Ket0())
    term = TensorPair(h_qubit, Ket0())
    print(f"Term: (H[|0⟩]) ⊗ |0⟩")
    
    # Type check
    tc = TypeChecker()
    typ = tc.check(term)
    print(f"Type: {typ}")
    
    # Analyze width
    wa = WidthAnalyzer()
    width = wa.analyze_term(term)
    print(f"Width: {width.width} qubits")
    
    # Extract circuit
    circuit = extract_circuit(term)
    print(f"Circuit: {circuit.size()} gates")
    
    return term


def example_quantum_control():
    """
    Example 3: Quantum Control with Orthogonal Branches
    
    qctrl(H[|0⟩], |0⟩, |1⟩) creates controlled superposition.
    """
    print("\n" + "=" * 50)
    print("Example 3: Quantum Control")
    print("=" * 50)
    
    control = UnitaryApp(UnitaryGate.H, Ket0())
    branch0 = Ket0()
    branch1 = Ket1()
    term = QCtrl(control, branch0, branch1)
    print(f"Term: qctrl(H[|0⟩], |0⟩, |1⟩)")
    
    # Verify orthogonality
    orth = OrthogonalityChecker()
    is_orth = orth.check_orthogonal(branch0, branch1)
    print(f"Branches orthogonal: {is_orth}")
    
    # Type check with orthogonality
    tc = TypeChecker(orthogonality_checker=orth)
    typ = tc.check(term)
    print(f"Type: {typ}")
    
    # Analyze
    ca = ComplexityAnalyzer()
    bound = ca.analyze_term(term)
    print(f"Complexity: {bound}")
    
    return term


def example_lambda_abstraction():
    """
    Example 4: Lambda Abstraction
    
    λx. H[x] : qubit ⊸ qubit
    """
    print("\n" + "=" * 50)
    print("Example 4: Lambda Abstraction")
    print("=" * 50)
    
    term = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
    print(f"Term: λx. H[x]")
    
    # Type check
    tc = TypeChecker()
    typ = tc.check(term)
    print(f"Type: {typ}")
    
    return term


def example_application():
    """
    Example 5: Function Application
    
    (λx. H[x]) |0⟩ = H[|0⟩]
    """
    print("\n" + "=" * 50)
    print("Example 5: Function Application")
    print("=" * 50)
    
    func = Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))
    arg = Ket0()
    term = Application(func, arg)
    print(f"Term: (λx. H[x]) |0⟩")
    
    # Type check
    tc = TypeChecker()
    typ = tc.check(term)
    print(f"Type: {typ}")
    
    # Extract circuit
    circuit = extract_circuit(term)
    print(f"Circuit: {circuit.size()} gates")
    
    return term


def example_superposition():
    """
    Example 6: Explicit Superposition
    
    (1/√2)|0⟩ + (1/√2)|1⟩
    """
    print("\n" + "=" * 50)
    print("Example 6: Superposition")
    print("=" * 50)
    
    from qllr.core.syntax import Superposition
    sqrt2 = 1 / cmath.sqrt(2)
    term = Superposition(sqrt2, Ket0(), sqrt2, Ket1())
    print(f"Term: (1/√2)|0⟩ + (1/√2)|1⟩")
    print(f"Normalized: {term.is_normalized()}")
    
    # Type check
    tc = TypeChecker()
    typ = tc.check(term)
    print(f"Type: {typ}")
    
    return term


def example_measurement():
    """
    Example 7: Measurement
    
    meas_Z(H[|0⟩])
    """
    print("\n" + "=" * 50)
    print("Example 7: Measurement")
    print("=" * 50)
    
    term = Measurement(
        MeasurementBasis.COMPUTATIONAL,
        UnitaryApp(UnitaryGate.H, Ket0())
    )
    print(f"Term: meas_Z(H[|0⟩])")
    
    # Type check
    tc = TypeChecker()
    typ = tc.check(term)
    print(f"Type: {typ}")  # Should be bool
    
    return term


def example_let_tensor():
    """
    Example 8: Let-Tensor Elimination
    
    let x ⊗ y = (|0⟩ ⊗ |1⟩) in H[x]
    """
    print("\n" + "=" * 50)
    print("Example 8: Let-Tensor")
    print("=" * 50)
    
    tensor = TensorPair(Ket0(), Ket1())
    term = LetTensor("x", "y", tensor, UnitaryApp(UnitaryGate.H, Variable("x")))
    print(f"Term: let x ⊗ y = (|0⟩ ⊗ |1⟩) in H[x]")
    
    # Type check
    tc = TypeChecker()
    typ = tc.check(term)
    print(f"Type: {typ}")
    
    return term


def example_higher_order_orthogonal():
    """
    Example 9: Higher-Order Orthogonality
    
    Two functions that differ only in an orthogonal constant.
    """
    print("\n" + "=" * 50)
    print("Example 9: Higher-Order Orthogonality")
    print("=" * 50)
    
    # λx. (H[x] ⊗ |0⟩)
    f0 = Abstraction("x", TensorPair(
        UnitaryApp(UnitaryGate.H, Variable("x")),
        Ket0()
    ))
    
    # λx. (H[x] ⊗ |1⟩)
    f1 = Abstraction("x", TensorPair(
        UnitaryApp(UnitaryGate.H, Variable("x")),
        Ket1()
    ))
    
    print(f"f₀ = λx. (H[x] ⊗ |0⟩)")
    print(f"f₁ = λx. (H[x] ⊗ |1⟩)")
    
    # Check orthogonality
    orth = OrthogonalityChecker()
    is_orth = orth.check_orthogonal(f0, f1)
    print(f"f₀ ⊥ f₁: {is_orth}")
    
    return f0, f1


def example_complexity_analysis():
    """
    Example 10: Complexity Analysis with Paragraph Modality
    
    Demonstrates how modal depth affects complexity bounds.
    """
    print("\n" + "=" * 50)
    print("Example 10: Complexity Analysis")
    print("=" * 50)
    
    ca = ComplexityAnalyzer()
    
    # Depth 0
    t0 = Ket0()
    b0 = ca.analyze_term(t0)
    print(f"|0⟩: depth={ca._term_depth(t0)}, bound={b0}")
    
    # Depth 1
    t1 = ParagraphIntro(Ket0())
    b1 = ca.analyze_term(t1)
    print(f"§|0⟩: depth={ca._term_depth(t1)}, bound={b1}")
    
    # Depth 2
    t2 = ParagraphIntro(ParagraphIntro(Ket0()))
    b2 = ca.analyze_term(t2)
    print(f"§§|0⟩: depth={ca._term_depth(t2)}, bound={b2}")
    
    return t0, t1, t2


def run_all_examples():
    """Run all examples."""
    example_hadamard()
    example_bell_state()
    example_quantum_control()
    example_lambda_abstraction()
    example_application()
    example_superposition()
    example_measurement()
    example_let_tensor()
    example_higher_order_orthogonal()
    example_complexity_analysis()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_examples()
