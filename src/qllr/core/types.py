"""
Type System for λ^QLLR

Defines the type grammar for the Quantum Light Linear Realizability calculus:
A, B ::= qubit | A ⊸ B | A ⊗ B | A ⊕ B | !A | §A | ♯A | ∀X.A

Based on Definition 3.1 from the research proposal.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set, Dict


class Type(ABC):
    """Abstract base class for all λ^QLLR types."""
    
    @abstractmethod
    def free_type_variables(self) -> Set[str]:
        """Return the set of free type variables."""
        pass
    
    @abstractmethod
    def substitute_type(self, var: str, replacement: 'Type') -> 'Type':
        """Substitute a type variable with a type."""
        pass
    
    @abstractmethod
    def modal_depth(self) -> int:
        """
        Compute the modal depth (nesting of § modalities).
        This determines the polynomial bound on reduction steps.
        """
        pass
    
    @abstractmethod
    def is_linear(self) -> bool:
        """
        Check if this type requires linear usage.
        Types not under ! or at depth 0 are linear.
        """
        pass
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass


@dataclass(frozen=True)
class QubitType(Type):
    """
    Qubit type: qubit
    
    Base type for quantum data. Linear by default (no-cloning).
    Width = 1 qubit.
    """
    
    def free_type_variables(self) -> Set[str]:
        return set()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return self
    
    def modal_depth(self) -> int:
        return 0
    
    def is_linear(self) -> bool:
        return True
    
    def width(self) -> int:
        """Return the qubit width (space requirement)."""
        return 1
    
    def __str__(self) -> str:
        return "qubit"
    
    def __repr__(self) -> str:
        return "QubitType()"


@dataclass(frozen=True)
class UnitType(Type):
    """Unit type: 1 (multiplicative unit)"""
    
    def free_type_variables(self) -> Set[str]:
        return set()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return self
    
    def modal_depth(self) -> int:
        return 0
    
    def is_linear(self) -> bool:
        return False  # Unit can be discarded
    
    def __str__(self) -> str:
        return "1"
    
    def __repr__(self) -> str:
        return "UnitType()"


@dataclass(frozen=True)
class BoolType(Type):
    """Boolean type: bool (classical data)"""
    
    def free_type_variables(self) -> Set[str]:
        return set()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return self
    
    def modal_depth(self) -> int:
        return 0
    
    def is_linear(self) -> bool:
        return False  # Classical data can be copied
    
    def __str__(self) -> str:
        return "bool"
    
    def __repr__(self) -> str:
        return "BoolType()"


@dataclass(frozen=True)
class TypeVariable(Type):
    """Type variable: X (for polymorphism)"""
    name: str
    
    def free_type_variables(self) -> Set[str]:
        return {self.name}
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        if self.name == var:
            return replacement
        return self
    
    def modal_depth(self) -> int:
        return 0
    
    def is_linear(self) -> bool:
        return True  # Conservative: assume linear
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"TypeVariable({self.name!r})"


@dataclass(frozen=True)
class LinearArrow(Type):
    """
    Linear function type: A ⊸ B
    
    Functions that use their argument exactly once.
    """
    domain: Type
    codomain: Type
    
    def free_type_variables(self) -> Set[str]:
        return self.domain.free_type_variables() | self.codomain.free_type_variables()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return LinearArrow(
            self.domain.substitute_type(var, replacement),
            self.codomain.substitute_type(var, replacement)
        )
    
    def modal_depth(self) -> int:
        return max(self.domain.modal_depth(), self.codomain.modal_depth())
    
    def is_linear(self) -> bool:
        return True
    
    def width(self) -> int:
        """Space requirement: max of domain and codomain (sequential execution)."""
        return max(self._get_width(self.domain), self._get_width(self.codomain))
    
    @staticmethod
    def _get_width(t: Type) -> int:
        if hasattr(t, 'width'):
            return t.width()
        return 0
    
    def __str__(self) -> str:
        return f"({self.domain} ⊸ {self.codomain})"
    
    def __repr__(self) -> str:
        return f"LinearArrow({self.domain!r}, {self.codomain!r})"


@dataclass(frozen=True)
class TensorProduct(Type):
    """
    Tensor product type: A ⊗ B
    
    Represents simultaneous availability of both types.
    """
    left: Type
    right: Type
    
    def free_type_variables(self) -> Set[str]:
        return self.left.free_type_variables() | self.right.free_type_variables()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return TensorProduct(
            self.left.substitute_type(var, replacement),
            self.right.substitute_type(var, replacement)
        )
    
    def modal_depth(self) -> int:
        return max(self.left.modal_depth(), self.right.modal_depth())
    
    def is_linear(self) -> bool:
        return self.left.is_linear() or self.right.is_linear()
    
    def width(self) -> int:
        """Space requirement: sum of both (held simultaneously)."""
        left_w = self.left.width() if hasattr(self.left, 'width') else 0
        right_w = self.right.width() if hasattr(self.right, 'width') else 0
        return left_w + right_w
    
    def __str__(self) -> str:
        return f"({self.left} ⊗ {self.right})"
    
    def __repr__(self) -> str:
        return f"TensorProduct({self.left!r}, {self.right!r})"


@dataclass(frozen=True)
class SumType(Type):
    """
    Sum type: A ⊕ B
    
    Represents choice between two types.
    """
    left: Type
    right: Type
    
    def free_type_variables(self) -> Set[str]:
        return self.left.free_type_variables() | self.right.free_type_variables()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return SumType(
            self.left.substitute_type(var, replacement),
            self.right.substitute_type(var, replacement)
        )
    
    def modal_depth(self) -> int:
        return max(self.left.modal_depth(), self.right.modal_depth())
    
    def is_linear(self) -> bool:
        return self.left.is_linear() or self.right.is_linear()
    
    def width(self) -> int:
        """Space requirement: max of both (only one active)."""
        left_w = self.left.width() if hasattr(self.left, 'width') else 0
        right_w = self.right.width() if hasattr(self.right, 'width') else 0
        return max(left_w, right_w)
    
    def __str__(self) -> str:
        return f"({self.left} ⊕ {self.right})"
    
    def __repr__(self) -> str:
        return f"SumType({self.left!r}, {self.right!r})"


@dataclass(frozen=True)
class BangType(Type):
    """
    Bang modality: !A ("of course")
    
    Marks data that can be duplicated without restriction.
    In DLAL, this is only at depth 0 for polynomial time.
    """
    inner: Type
    
    def free_type_variables(self) -> Set[str]:
        return self.inner.free_type_variables()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return BangType(self.inner.substitute_type(var, replacement))
    
    def modal_depth(self) -> int:
        # ! does not increase depth in DLAL
        return self.inner.modal_depth()
    
    def is_linear(self) -> bool:
        return False  # Can be duplicated
    
    def width(self) -> int:
        """Same width as inner type."""
        return self.inner.width() if hasattr(self.inner, 'width') else 0
    
    def __str__(self) -> str:
        return f"!{self.inner}"
    
    def __repr__(self) -> str:
        return f"BangType({self.inner!r})"


@dataclass(frozen=True)
class ParagraphType(Type):
    """
    Paragraph modality: §A 
    
    Marks data with bounded duplication depth for polynomial time.
    Each nesting of § increases the modal depth by 1.
    """
    inner: Type
    
    def free_type_variables(self) -> Set[str]:
        return self.inner.free_type_variables()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return ParagraphType(self.inner.substitute_type(var, replacement))
    
    def modal_depth(self) -> int:
        # § increases depth by 1
        return 1 + self.inner.modal_depth()
    
    def is_linear(self) -> bool:
        return False  # Can be duplicated (bounded times)
    
    def width(self) -> int:
        """Same width as inner type."""
        return self.inner.width() if hasattr(self.inner, 'width') else 0
    
    def __str__(self) -> str:
        return f"§{self.inner}"
    
    def __repr__(self) -> str:
        return f"ParagraphType({self.inner!r})"


@dataclass(frozen=True)
class SharpType(Type):
    """
    Superposition modality: ♯A
    
    Marks types that may contain quantum superpositions.
    Linear due to no-cloning.
    """
    inner: Type
    
    def free_type_variables(self) -> Set[str]:
        return self.inner.free_type_variables()
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        return SharpType(self.inner.substitute_type(var, replacement))
    
    def modal_depth(self) -> int:
        return self.inner.modal_depth()
    
    def is_linear(self) -> bool:
        return True  # No-cloning for superpositions
    
    def width(self) -> int:
        """Same width as inner type."""
        return self.inner.width() if hasattr(self.inner, 'width') else 0
    
    def __str__(self) -> str:
        return f"♯{self.inner}"
    
    def __repr__(self) -> str:
        return f"SharpType({self.inner!r})"


@dataclass(frozen=True)
class ForallType(Type):
    """
    Universal quantification: ∀X.A
    
    Parametric polymorphism over types.
    """
    var: str
    body: Type
    
    def free_type_variables(self) -> Set[str]:
        return self.body.free_type_variables() - {self.var}
    
    def substitute_type(self, var: str, replacement: Type) -> Type:
        if var == self.var:
            return self
        if self.var in replacement.free_type_variables():
            # Need alpha-conversion
            new_var = self._fresh_var(self.var, replacement.free_type_variables() | self.body.free_type_variables())
            new_body = self.body.substitute_type(self.var, TypeVariable(new_var))
            return ForallType(new_var, new_body.substitute_type(var, replacement))
        return ForallType(self.var, self.body.substitute_type(var, replacement))
    
    @staticmethod
    def _fresh_var(base: str, avoid: Set[str]) -> str:
        i = 0
        while f"{base}{i}" in avoid:
            i += 1
        return f"{base}{i}"
    
    def modal_depth(self) -> int:
        return self.body.modal_depth()
    
    def is_linear(self) -> bool:
        return self.body.is_linear()
    
    def instantiate(self, typ: Type) -> Type:
        """Instantiate the quantified variable with a concrete type."""
        return self.body.substitute_type(self.var, typ)
    
    def __str__(self) -> str:
        return f"(∀{self.var}. {self.body})"
    
    def __repr__(self) -> str:
        return f"ForallType({self.var!r}, {self.body!r})"


# Type aliases for common patterns
QubitPair = TensorProduct(QubitType(), QubitType())
QubitToQubit = LinearArrow(QubitType(), QubitType())


def types_equal(t1: Type, t2: Type) -> bool:
    """Check structural equality of types (up to alpha-equivalence)."""
    if type(t1) != type(t2):
        return False
    
    if isinstance(t1, (QubitType, UnitType, BoolType)):
        return True
    
    if isinstance(t1, TypeVariable):
        return t1.name == t2.name
    
    if isinstance(t1, LinearArrow):
        return types_equal(t1.domain, t2.domain) and types_equal(t1.codomain, t2.codomain)
    
    if isinstance(t1, (TensorProduct, SumType)):
        return types_equal(t1.left, t2.left) and types_equal(t1.right, t2.right)
    
    if isinstance(t1, (BangType, ParagraphType, SharpType)):
        return types_equal(t1.inner, t2.inner)
    
    if isinstance(t1, ForallType):
        # Alpha-equivalence
        fresh = ForallType._fresh_var("X", t1.free_type_variables() | t2.free_type_variables())
        body1 = t1.body.substitute_type(t1.var, TypeVariable(fresh))
        body2 = t2.body.substitute_type(t2.var, TypeVariable(fresh))
        return types_equal(body1, body2)
    
    return False
