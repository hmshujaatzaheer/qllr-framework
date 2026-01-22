"""
Width (Space) Analysis for λ^QLLR

Implements space complexity certification via width-refined types,
following Section 6 of the research proposal.

Theorem 6.1 (Space Soundness): If Γ; Δ ⊢_w M : A^n, then executing M
requires at most n qubits simultaneously.

Key rules for width computation:
- qubit^1: 1 qubit
- (A^w₁ ⊸ B^w₂)^{max(w₁,w₂)}: sequential execution
- (A^w₁ ⊗ B^w₂)^{w₁+w₂}: parallel composition (held simultaneously)
- qctrl adds 1 for the control qubit
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from qllr.core.syntax import (
    Term, Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryApp, QCtrl, Measurement, New, Unit
)
from qllr.core.types import (
    Type, QubitType, UnitType, BoolType, LinearArrow, TensorProduct,
    SumType, BangType, ParagraphType, SharpType, ForallType
)


@dataclass
class WidthBound:
    """
    A bound on qubit width (space requirement).
    
    Attributes:
        width: Maximum number of qubits needed simultaneously
        peak_width: Peak width during execution (may be higher than final)
    """
    width: int
    peak_width: Optional[int] = None
    
    def __post_init__(self):
        if self.peak_width is None:
            self.peak_width = self.width
    
    def sequential(self, other: 'WidthBound') -> 'WidthBound':
        """Combine for sequential execution (max width)."""
        return WidthBound(
            width=max(self.width, other.width),
            peak_width=max(self.peak_width, other.peak_width)
        )
    
    def parallel(self, other: 'WidthBound') -> 'WidthBound':
        """Combine for parallel execution (sum of widths)."""
        return WidthBound(
            width=self.width + other.width,
            peak_width=self.peak_width + other.peak_width
        )
    
    def add_control(self) -> 'WidthBound':
        """Add a control qubit (for quantum control)."""
        return WidthBound(
            width=self.width + 1,
            peak_width=self.peak_width + 1
        )
    
    def __str__(self) -> str:
        if self.peak_width == self.width:
            return f"width={self.width}"
        return f"width={self.width}, peak={self.peak_width}"
    
    def __repr__(self) -> str:
        return f"WidthBound({self.width}, {self.peak_width})"


class WidthAnalyzer:
    """
    Analyzes space (qubit width) requirements of λ^QLLR terms.
    
    Implements the width-refined type system from Section 6:
    - A^n means type A requiring at most n qubits
    - Width typing judgment: Γ; Δ ⊢_w M : A^n
    
    Key properties:
    - Qubit operations: width 1
    - Tensor products: sum of widths (parallel)
    - Functions: max of domain and codomain (sequential)
    - Quantum control: branches + 1 for control qubit
    """
    
    def analyze_type(self, typ: Type) -> WidthBound:
        """
        Compute width bound from a type.
        
        Width rules:
        - qubit: 1
        - A ⊸ B: max(width(A), width(B))
        - A ⊗ B: width(A) + width(B)
        - A ⊕ B: max(width(A), width(B))
        - !A, §A, ♯A: width(A)
        """
        if isinstance(typ, QubitType):
            return WidthBound(1)
        
        if isinstance(typ, (UnitType, BoolType)):
            return WidthBound(0)
        
        if isinstance(typ, LinearArrow):
            domain_width = self.analyze_type(typ.domain)
            codomain_width = self.analyze_type(typ.codomain)
            return domain_width.sequential(codomain_width)
        
        if isinstance(typ, TensorProduct):
            left_width = self.analyze_type(typ.left)
            right_width = self.analyze_type(typ.right)
            return left_width.parallel(right_width)
        
        if isinstance(typ, SumType):
            left_width = self.analyze_type(typ.left)
            right_width = self.analyze_type(typ.right)
            return left_width.sequential(right_width)  # Max (only one active)
        
        if isinstance(typ, (BangType, ParagraphType, SharpType)):
            return self.analyze_type(typ.inner)
        
        if isinstance(typ, ForallType):
            return self.analyze_type(typ.body)
        
        return WidthBound(0)
    
    def analyze_term(self, term: Term, context: Optional[Dict[str, int]] = None) -> WidthBound:
        """
        Analyze the width (space requirement) of a term.
        
        Returns the maximum number of qubits needed during execution.
        """
        if context is None:
            context = {}
        
        return self._analyze(term, context)
    
    def _analyze(self, term: Term, ctx: Dict[str, int]) -> WidthBound:
        """Internal analysis with variable context."""
        
        if isinstance(term, (Ket0, Ket1)):
            return WidthBound(1)
        
        if isinstance(term, New):
            return WidthBound(1)
        
        if isinstance(term, Unit):
            return WidthBound(0)
        
        if isinstance(term, Variable):
            # Variable contributes its assigned width
            return WidthBound(ctx.get(term.name, 1))
        
        if isinstance(term, Superposition):
            # Superposition: both terms share the same qubits
            w0 = self._analyze(term.term0, ctx)
            w1 = self._analyze(term.term1, ctx)
            # Same qubits, different amplitudes
            return w0.sequential(w1)
        
        if isinstance(term, Abstraction):
            # Add parameter with width 1 (qubit)
            new_ctx = {**ctx, term.var: 1}
            body_width = self._analyze(term.body, new_ctx)
            return body_width
        
        if isinstance(term, Application):
            func_width = self._analyze(term.func, ctx)
            arg_width = self._analyze(term.arg, ctx)
            # Sequential: argument first, then function body
            return func_width.sequential(arg_width)
        
        if isinstance(term, TensorPair):
            left_width = self._analyze(term.left, ctx)
            right_width = self._analyze(term.right, ctx)
            # Parallel: both held simultaneously
            return left_width.parallel(right_width)
        
        if isinstance(term, LetTensor):
            tensor_width = self._analyze(term.tensor_term, ctx)
            # Add both variables
            new_ctx = {**ctx, term.var_left: 1, term.var_right: 1}
            body_width = self._analyze(term.body, new_ctx)
            return tensor_width.sequential(body_width)
        
        if isinstance(term, (BangIntro, ParagraphIntro)):
            return self._analyze(term.term, ctx)
        
        if isinstance(term, LetBang):
            bang_width = self._analyze(term.bang_term, ctx)
            new_ctx = {**ctx, term.var: bang_width.width}
            body_width = self._analyze(term.body, new_ctx)
            return bang_width.sequential(body_width)
        
        if isinstance(term, UnitaryApp):
            arg_width = self._analyze(term.arg, ctx)
            # Unitary preserves width (same qubits in/out)
            return arg_width
        
        if isinstance(term, QCtrl):
            ctrl_width = self._analyze(term.control, ctx)
            b0_width = self._analyze(term.branch0, ctx)
            b1_width = self._analyze(term.branch1, ctx)
            # Control qubit + max of branches
            branches = b0_width.sequential(b1_width)
            return ctrl_width.parallel(branches).add_control()
        
        if isinstance(term, Measurement):
            return self._analyze(term.arg, ctx)
        
        return WidthBound(0)
    
    def verify_width_bound(self, term: Term, bound: int) -> Tuple[bool, WidthBound]:
        """
        Verify that a term respects a width bound.
        
        Returns (satisfies_bound, actual_width).
        """
        actual = self.analyze_term(term)
        return (actual.width <= bound, actual)
    
    def compute_circuit_width(self, term: Term) -> int:
        """
        Compute the width of the extracted circuit.
        
        This is the minimum number of qubits needed in the circuit.
        """
        return self.analyze_term(term).width
    
    def estimate_ancilla_count(self, term: Term) -> int:
        """
        Estimate the number of ancilla qubits needed.
        
        Ancillas are needed for:
        - Quantum control (1 per qctrl)
        - Uncomputation
        """
        return self._count_ancillas(term)
    
    def _count_ancillas(self, term: Term) -> int:
        """Count ancilla requirements recursively."""
        if isinstance(term, (Variable, Ket0, Ket1, New, Unit)):
            return 0
        
        if isinstance(term, Superposition):
            return max(self._count_ancillas(term.term0),
                      self._count_ancillas(term.term1))
        
        if isinstance(term, Abstraction):
            return self._count_ancillas(term.body)
        
        if isinstance(term, Application):
            return (self._count_ancillas(term.func) + 
                   self._count_ancillas(term.arg))
        
        if isinstance(term, TensorPair):
            return (self._count_ancillas(term.left) + 
                   self._count_ancillas(term.right))
        
        if isinstance(term, LetTensor):
            return (self._count_ancillas(term.tensor_term) + 
                   self._count_ancillas(term.body))
        
        if isinstance(term, (BangIntro, ParagraphIntro)):
            return self._count_ancillas(term.term)
        
        if isinstance(term, LetBang):
            return (self._count_ancillas(term.bang_term) + 
                   self._count_ancillas(term.body))
        
        if isinstance(term, UnitaryApp):
            return self._count_ancillas(term.arg)
        
        if isinstance(term, QCtrl):
            # Each qctrl needs 1 ancilla for control
            return (1 + self._count_ancillas(term.control) +
                   max(self._count_ancillas(term.branch0),
                       self._count_ancillas(term.branch1)))
        
        if isinstance(term, Measurement):
            return self._count_ancillas(term.arg)
        
        return 0


def analyze_width(term: Term) -> WidthBound:
    """Convenience function to analyze term width."""
    analyzer = WidthAnalyzer()
    return analyzer.analyze_term(term)


def compute_qubit_count(term: Term) -> int:
    """Convenience function to compute qubit count."""
    analyzer = WidthAnalyzer()
    return analyzer.compute_circuit_width(term)
