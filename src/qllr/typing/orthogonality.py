"""
Orthogonality Checker for Higher-Order Terms

Implements Algorithm 3 (Higher-Order Orthogonality Checking) from the 
research proposal.

The key insight is that for efficient circuit compilation via 
anchoring-and-merging, the branches of quantum control must be orthogonal.
For first-order terms, this means they differ at positions with orthogonal
basis states. For higher-order terms (functions), we use a syntactic
criterion that is sound but conservative.

Definitions from the proposal:
- Definition 3.5: Higher-Order Orthogonality (semantic)
- Definition 3.6: Syntactic Orthogonality Criterion
- Proposition 3.1: Soundness of Syntactic Criterion
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from enum import Enum

from qllr.core.syntax import (
    Term, Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, Measurement, New, Unit, Inl, Inr, Case
)


@dataclass
class Difference:
    """Represents a structural difference between two terms."""
    position: List[str]  # Path to the differing subterms
    term0: Term
    term1: Term
    
    def involves_bound_var(self, bound_vars: Set[str]) -> bool:
        """Check if the difference involves any bound variables."""
        fv0 = self.term0.free_variables()
        fv1 = self.term1.free_variables()
        return bool((fv0 | fv1) & bound_vars)


class OrthogonalityChecker:
    """
    Checker for orthogonality of terms.
    
    Implements both first-order orthogonality (for basis states) and
    higher-order orthogonality (for functions) using the syntactic criterion.
    
    Algorithm 3 from the proposal:
    1. Find structural differences between terms
    2. Check that differences don't involve bound variables
    3. Verify each difference is first-order orthogonal
    """
    
    def check_orthogonal(self, term0: Term, term1: Term) -> bool:
        """
        Check if two terms are orthogonal.
        
        For basis states: |0⟩ ⊥ |1⟩
        For functions: λx.M₀ ⊥ λx.M₁ if syntactic criterion is satisfied
        
        Args:
            term0: First term
            term1: Second term
            
        Returns:
            True if terms are orthogonal
        """
        # Check for identical terms (not orthogonal)
        if self._structurally_equal(term0, term1):
            return False
        
        # Check if both are abstractions (higher-order case)
        if isinstance(term0, Abstraction) and isinstance(term1, Abstraction):
            return self._check_higher_order_orthogonal(term0, term1)
        
        # First-order case
        return self._check_first_order_orthogonal(term0, term1)
    
    def _check_first_order_orthogonal(self, term0: Term, term1: Term) -> bool:
        """
        Check first-order orthogonality.
        
        Two first-order terms are orthogonal if they are distinguishable
        basis states or contain orthogonal subterms at the same position.
        
        Definition 3.4: Terms N₀, N₁ : A are orthogonal if there exists
        a unitary U and basis states |φ₀⟩, |φ₁⟩ such that
        ⟦Nᵢ⟧ = U · (· ⊗ |φᵢ⟩) with ⟨φ₀|φ₁⟩ = 0.
        """
        # Base case: basis states
        if isinstance(term0, Ket0) and isinstance(term1, Ket1):
            return True
        if isinstance(term0, Ket1) and isinstance(term1, Ket0):
            return True
        
        # Superpositions with orthogonal components
        if isinstance(term0, Superposition) and isinstance(term1, Superposition):
            # Check if they span orthogonal subspaces
            # Simplified: check if base terms are orthogonal
            if self._check_first_order_orthogonal(term0.term0, term1.term0):
                if self._check_first_order_orthogonal(term0.term1, term1.term1):
                    return True
        
        # Tensor products: orthogonal if any component is orthogonal
        if isinstance(term0, TensorPair) and isinstance(term1, TensorPair):
            if self._check_first_order_orthogonal(term0.left, term1.left):
                return True
            if self._check_first_order_orthogonal(term0.right, term1.right):
                return True
        
        # Unitary applications: orthogonality preserved by unitary
        if isinstance(term0, UnitaryApp) and isinstance(term1, UnitaryApp):
            if term0.gate == term1.gate:
                return self._check_first_order_orthogonal(term0.arg, term1.arg)
        
        # Applications that differ only in argument
        if isinstance(term0, Application) and isinstance(term1, Application):
            if self._structurally_equal(term0.func, term1.func):
                return self._check_first_order_orthogonal(term0.arg, term1.arg)
        
        return False
    
    def _check_higher_order_orthogonal(self, term0: Abstraction, term1: Abstraction) -> bool:
        """
        Check higher-order orthogonality using syntactic criterion.
        
        Definition 3.6 (Syntactic Orthogonality Criterion):
        Terms N₀ = λx.M₀ and N₁ = λx.M₁ satisfy the criterion if:
        1. M₀ and M₁ have identical structure except at orthogonal positions
        2. At each orthogonal position, subterms form a first-order orthogonal pair
        3. Orthogonal positions do not depend on x (the bound variable)
        
        Algorithm 3: CheckOrthogonal(N₀, N₁)
        """
        # Ensure same variable name (can rename if needed)
        if term0.var != term1.var:
            # Alpha-rename term1 to use same variable
            fresh_var = term0.var
            body1 = term1.body.substitute(term1.var, Variable(fresh_var))
        else:
            body1 = term1.body
        
        body0 = term0.body
        bound_var = term0.var
        
        # Find differences between bodies
        differences = self._find_differences(body0, body1, [])
        
        if not differences:
            # Identical bodies - not orthogonal
            return False
        
        # Check each difference
        for diff in differences:
            # Condition 3: difference must not depend on bound variable
            if diff.involves_bound_var({bound_var}):
                return False
            
            # Condition 2: difference must be first-order orthogonal
            if not self._check_first_order_orthogonal(diff.term0, diff.term1):
                return False
        
        return True
    
    def _find_differences(
        self, 
        term0: Term, 
        term1: Term, 
        path: List[str]
    ) -> List[Difference]:
        """
        Find all structural differences between two terms.
        
        Returns a list of Difference objects indicating where and how
        the terms differ.
        """
        # Same type check
        if type(term0) != type(term1):
            return [Difference(path.copy(), term0, term1)]
        
        # Base cases - values
        if isinstance(term0, (Ket0, Ket1, New, Unit)):
            if term0 != term1:
                return [Difference(path.copy(), term0, term1)]
            return []
        
        if isinstance(term0, Variable):
            if term0.name != term1.name:
                return [Difference(path.copy(), term0, term1)]
            return []
        
        # Recursive cases
        if isinstance(term0, Superposition):
            diffs = []
            if term0.alpha != term1.alpha or term0.beta != term1.beta:
                diffs.append(Difference(path + ["coeff"], term0, term1))
            diffs.extend(self._find_differences(term0.term0, term1.term0, path + ["0"]))
            diffs.extend(self._find_differences(term0.term1, term1.term1, path + ["1"]))
            return diffs
        
        if isinstance(term0, Abstraction):
            if term0.var != term1.var:
                return [Difference(path.copy(), term0, term1)]
            return self._find_differences(term0.body, term1.body, path + ["body"])
        
        if isinstance(term0, Application):
            diffs = []
            diffs.extend(self._find_differences(term0.func, term1.func, path + ["func"]))
            diffs.extend(self._find_differences(term0.arg, term1.arg, path + ["arg"]))
            return diffs
        
        if isinstance(term0, TensorPair):
            diffs = []
            diffs.extend(self._find_differences(term0.left, term1.left, path + ["left"]))
            diffs.extend(self._find_differences(term0.right, term1.right, path + ["right"]))
            return diffs
        
        if isinstance(term0, UnitaryApp):
            if term0.gate != term1.gate:
                return [Difference(path + ["gate"], term0, term1)]
            return self._find_differences(term0.arg, term1.arg, path + ["arg"])
        
        if isinstance(term0, QCtrl):
            diffs = []
            diffs.extend(self._find_differences(term0.control, term1.control, path + ["ctrl"]))
            diffs.extend(self._find_differences(term0.branch0, term1.branch0, path + ["b0"]))
            diffs.extend(self._find_differences(term0.branch1, term1.branch1, path + ["b1"]))
            return diffs
        
        if isinstance(term0, LetTensor):
            diffs = []
            if term0.var_left != term1.var_left or term0.var_right != term1.var_right:
                diffs.append(Difference(path + ["vars"], term0, term1))
            diffs.extend(self._find_differences(term0.tensor_term, term1.tensor_term, path + ["tensor"]))
            diffs.extend(self._find_differences(term0.body, term1.body, path + ["body"]))
            return diffs
        
        if isinstance(term0, BangIntro):
            return self._find_differences(term0.term, term1.term, path + ["inner"])
        
        if isinstance(term0, LetBang):
            diffs = []
            if term0.var != term1.var:
                diffs.append(Difference(path + ["var"], term0, term1))
            diffs.extend(self._find_differences(term0.bang_term, term1.bang_term, path + ["bang"]))
            diffs.extend(self._find_differences(term0.body, term1.body, path + ["body"]))
            return diffs
        
        if isinstance(term0, ParagraphIntro):
            return self._find_differences(term0.term, term1.term, path + ["inner"])
        
        if isinstance(term0, Measurement):
            if term0.basis != term1.basis:
                return [Difference(path + ["basis"], term0, term1)]
            return self._find_differences(term0.arg, term1.arg, path + ["arg"])
        
        if isinstance(term0, (Inl, Inr)):
            return self._find_differences(term0.term, term1.term, path + ["inner"])
        
        if isinstance(term0, Case):
            diffs = []
            diffs.extend(self._find_differences(term0.scrutinee, term1.scrutinee, path + ["scrut"]))
            diffs.extend(self._find_differences(term0.branch_left, term1.branch_left, path + ["left"]))
            diffs.extend(self._find_differences(term0.branch_right, term1.branch_right, path + ["right"]))
            return diffs
        
        # Fallback
        return [Difference(path.copy(), term0, term1)]
    
    def _structurally_equal(self, term0: Term, term1: Term) -> bool:
        """Check if two terms are structurally equal."""
        return not self._find_differences(term0, term1, [])


def check_orthogonality(term0: Term, term1: Term) -> bool:
    """
    Convenience function to check orthogonality.
    
    Example usage:
    >>> check_orthogonality(Ket0(), Ket1())
    True
    >>> check_orthogonality(Ket0(), Ket0())
    False
    """
    checker = OrthogonalityChecker()
    return checker.check_orthogonal(term0, term1)
