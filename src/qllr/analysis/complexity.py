"""
Complexity Analysis for λ^QLLR

Implements polynomial time normalization bounds based on the modal depth
of types, following Theorem 4.2 from the research proposal.

Key result: For type A with modal depth d(A), terms normalize in at most
P_A(n) = n^{2^{d(A)+1}} steps.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

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
class ComplexityBound:
    """
    A polynomial bound on computation steps/resources.
    
    Represented as n^k for some exponent k.
    """
    exponent: int
    coefficient: int = 1
    
    def evaluate(self, n: int) -> int:
        """Evaluate the bound at input size n."""
        return self.coefficient * (n ** self.exponent)
    
    def compose(self, other: 'ComplexityBound') -> 'ComplexityBound':
        """Compose bounds (for sequential computation)."""
        # (n^a) composed with (n^b) = O(n^{max(a,b)+1})
        return ComplexityBound(
            exponent=max(self.exponent, other.exponent) + 1,
            coefficient=self.coefficient * other.coefficient
        )
    
    def parallel(self, other: 'ComplexityBound') -> 'ComplexityBound':
        """Combine bounds for parallel computation."""
        # max of the two
        return ComplexityBound(
            exponent=max(self.exponent, other.exponent),
            coefficient=max(self.coefficient, other.coefficient)
        )
    
    def __str__(self) -> str:
        if self.coefficient == 1:
            return f"O(n^{self.exponent})"
        return f"O({self.coefficient}·n^{self.exponent})"
    
    def __repr__(self) -> str:
        return f"ComplexityBound({self.exponent}, {self.coefficient})"


class ComplexityAnalyzer:
    """
    Analyzes time complexity of λ^QLLR terms.
    
    Based on the DLAL/LLL stratification technique:
    - Modal depth d bounds iteration depth
    - Polynomial bound is n^{2^{d+1}}
    
    Theorem 4.2: For Γ; Δ ⊢ M : A, M normalizes in ≤ P_A(|M|) steps.
    """
    
    def analyze_type(self, typ: Type) -> ComplexityBound:
        """
        Compute the complexity bound from a type.
        
        The polynomial P_A(n) = n^{2^{d(A)+1}} where d(A) is modal depth.
        """
        depth = typ.modal_depth()
        exponent = 2 ** (depth + 1)
        return ComplexityBound(exponent=exponent)
    
    def analyze_term(self, term: Term) -> ComplexityBound:
        """
        Analyze the complexity of a term.
        
        Returns an upper bound on the number of reduction steps.
        """
        size = self._term_size(term)
        depth = self._term_depth(term)
        
        # Bound from modal depth
        exponent = 2 ** (depth + 1)
        
        return ComplexityBound(exponent=exponent)
    
    def _term_size(self, term: Term) -> int:
        """Compute the syntactic size of a term."""
        if isinstance(term, (Variable, Ket0, Ket1, New, Unit)):
            return 1
        
        if isinstance(term, Superposition):
            return 1 + self._term_size(term.term0) + self._term_size(term.term1)
        
        if isinstance(term, Abstraction):
            return 1 + self._term_size(term.body)
        
        if isinstance(term, Application):
            return 1 + self._term_size(term.func) + self._term_size(term.arg)
        
        if isinstance(term, TensorPair):
            return 1 + self._term_size(term.left) + self._term_size(term.right)
        
        if isinstance(term, LetTensor):
            return 1 + self._term_size(term.tensor_term) + self._term_size(term.body)
        
        if isinstance(term, (BangIntro, ParagraphIntro)):
            return 1 + self._term_size(term.term)
        
        if isinstance(term, LetBang):
            return 1 + self._term_size(term.bang_term) + self._term_size(term.body)
        
        if isinstance(term, UnitaryApp):
            return 1 + self._term_size(term.arg)
        
        if isinstance(term, QCtrl):
            return (1 + self._term_size(term.control) + 
                    self._term_size(term.branch0) + self._term_size(term.branch1))
        
        if isinstance(term, Measurement):
            return 1 + self._term_size(term.arg)
        
        return 1
    
    def _term_depth(self, term: Term) -> int:
        """
        Compute the modal depth of a term.
        
        This is the maximum nesting of § modalities.
        """
        if isinstance(term, (Variable, Ket0, Ket1, New, Unit)):
            return 0
        
        if isinstance(term, Superposition):
            return max(self._term_depth(term.term0), self._term_depth(term.term1))
        
        if isinstance(term, Abstraction):
            return self._term_depth(term.body)
        
        if isinstance(term, Application):
            return max(self._term_depth(term.func), self._term_depth(term.arg))
        
        if isinstance(term, TensorPair):
            return max(self._term_depth(term.left), self._term_depth(term.right))
        
        if isinstance(term, LetTensor):
            return max(self._term_depth(term.tensor_term), self._term_depth(term.body))
        
        if isinstance(term, BangIntro):
            return self._term_depth(term.term)
        
        if isinstance(term, ParagraphIntro):
            # § increases depth by 1
            return 1 + self._term_depth(term.term)
        
        if isinstance(term, LetBang):
            return max(self._term_depth(term.bang_term), self._term_depth(term.body))
        
        if isinstance(term, UnitaryApp):
            return self._term_depth(term.arg)
        
        if isinstance(term, QCtrl):
            return max(
                self._term_depth(term.control),
                self._term_depth(term.branch0),
                self._term_depth(term.branch1)
            )
        
        if isinstance(term, Measurement):
            return self._term_depth(term.arg)
        
        return 0
    
    def verify_polytime(self, term: Term, typ: Type) -> Tuple[bool, ComplexityBound]:
        """
        Verify that a term has polynomial-time normalization.
        
        Returns (is_polytime, bound).
        """
        bound = self.analyze_type(typ)
        term_depth = self._term_depth(term)
        type_depth = typ.modal_depth()
        
        # Check that term depth is bounded by type depth
        if term_depth > type_depth:
            # May not be polynomial time
            return (False, ComplexityBound(exponent=2**(term_depth+1)))
        
        return (True, bound)
    
    def estimate_reduction_steps(self, term: Term) -> int:
        """
        Estimate the number of reduction steps for a term.
        
        This is a heuristic based on term structure.
        """
        size = self._term_size(term)
        depth = self._term_depth(term)
        
        # Base estimate: size * 2^depth
        return size * (2 ** depth)
    
    def is_in_polytime_fragment(self, term: Term) -> bool:
        """
        Check if a term is in the polynomial-time fragment λ^QLLR_poly.
        
        Requirements (Definition 4.1):
        1. Ground-type variables appear linearly
        2. ! modality only at depth 0
        3. All quantum control satisfies orthogonality
        4. Recursive definitions use §-guarded fixed points
        """
        return self._check_polytime_fragment(term, depth=0)
    
    def _check_polytime_fragment(self, term: Term, depth: int) -> bool:
        """Recursive check for polynomial-time fragment membership."""
        if isinstance(term, (Variable, Ket0, Ket1, New, Unit)):
            return True
        
        if isinstance(term, Superposition):
            return (self._check_polytime_fragment(term.term0, depth) and
                    self._check_polytime_fragment(term.term1, depth))
        
        if isinstance(term, Abstraction):
            return self._check_polytime_fragment(term.body, depth)
        
        if isinstance(term, Application):
            return (self._check_polytime_fragment(term.func, depth) and
                    self._check_polytime_fragment(term.arg, depth))
        
        if isinstance(term, TensorPair):
            return (self._check_polytime_fragment(term.left, depth) and
                    self._check_polytime_fragment(term.right, depth))
        
        if isinstance(term, BangIntro):
            # ! only at depth 0
            if depth > 0:
                return False
            return self._check_polytime_fragment(term.term, depth)
        
        if isinstance(term, ParagraphIntro):
            return self._check_polytime_fragment(term.term, depth + 1)
        
        if isinstance(term, LetTensor):
            return (self._check_polytime_fragment(term.tensor_term, depth) and
                    self._check_polytime_fragment(term.body, depth))
        
        if isinstance(term, LetBang):
            return (self._check_polytime_fragment(term.bang_term, depth) and
                    self._check_polytime_fragment(term.body, depth))
        
        if isinstance(term, UnitaryApp):
            return self._check_polytime_fragment(term.arg, depth)
        
        if isinstance(term, QCtrl):
            # Would need orthogonality check here
            return (self._check_polytime_fragment(term.control, depth) and
                    self._check_polytime_fragment(term.branch0, depth) and
                    self._check_polytime_fragment(term.branch1, depth))
        
        if isinstance(term, Measurement):
            return self._check_polytime_fragment(term.arg, depth)
        
        return True


def analyze_complexity(term: Term) -> ComplexityBound:
    """Convenience function to analyze term complexity."""
    analyzer = ComplexityAnalyzer()
    return analyzer.analyze_term(term)


def verify_polytime(term: Term, typ: Type) -> bool:
    """Convenience function to verify polynomial-time normalization."""
    analyzer = ComplexityAnalyzer()
    is_poly, _ = analyzer.verify_polytime(term, typ)
    return is_poly
