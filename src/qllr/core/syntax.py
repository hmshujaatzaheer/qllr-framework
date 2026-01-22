"""
Core Syntax Module for λ^QLLR

Defines the Abstract Syntax Tree (AST) for the Quantum Light Linear 
Realizability calculus with coherent control.

Based on Definition 2.1 from the research proposal:
- Types: qubit | A ⊸ B | A ⊗ B | A ⊕ B | !A | §A | ♯A | ∀X.A
- Terms: Values, Applications, Let bindings, Unitary operations, 
         Quantum control, Measurement
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set, FrozenSet
from enum import Enum
import cmath
import copy


class Term(ABC):
    """Abstract base class for all λ^QLLR terms."""
    
    @abstractmethod
    def free_variables(self) -> Set[str]:
        """Return the set of free variables in this term."""
        pass
    
    @abstractmethod
    def substitute(self, var: str, replacement: 'Term') -> 'Term':
        """Substitute a variable with a term."""
        pass
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    def is_value(self) -> bool:
        """Check if this term is a value (cannot be reduced further)."""
        return False
    
    def clone(self) -> 'Term':
        """Create a deep copy of this term."""
        return copy.deepcopy(self)


@dataclass(frozen=True)
class Variable(Term):
    """Variable term: x"""
    name: str
    
    def free_variables(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, replacement: Term) -> Term:
        if self.name == var:
            return replacement.clone()
        return self
    
    def is_value(self) -> bool:
        return True
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Variable({self.name!r})"


@dataclass(frozen=True)
class Ket0(Term):
    """Basis state |0⟩"""
    
    def free_variables(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return self
    
    def is_value(self) -> bool:
        return True
    
    def __str__(self) -> str:
        return "|0⟩"
    
    def __repr__(self) -> str:
        return "Ket0()"


@dataclass(frozen=True)
class Ket1(Term):
    """Basis state |1⟩"""
    
    def free_variables(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return self
    
    def is_value(self) -> bool:
        return True
    
    def __str__(self) -> str:
        return "|1⟩"
    
    def __repr__(self) -> str:
        return "Ket1()"


# Convenience aliases
Qubit = Ket0 | Ket1


@dataclass(frozen=True)
class Superposition(Term):
    """
    Linear combination of terms: α·t + β·r
    
    Represents quantum superposition where α, β are complex amplitudes
    satisfying |α|² + |β|² = 1 for normalized states.
    """
    alpha: complex
    term0: Term
    beta: complex
    term1: Term
    
    def __post_init__(self):
        # Validate normalization (with tolerance for floating point)
        norm_sq = abs(self.alpha)**2 + abs(self.beta)**2
        if not (0.99 <= norm_sq <= 1.01):
            # Allow unnormalized for intermediate computations
            pass
    
    def free_variables(self) -> Set[str]:
        return self.term0.free_variables() | self.term1.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return Superposition(
            self.alpha,
            self.term0.substitute(var, replacement),
            self.beta,
            self.term1.substitute(var, replacement)
        )
    
    def is_value(self) -> bool:
        return self.term0.is_value() and self.term1.is_value()
    
    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """Check if amplitudes are normalized."""
        norm_sq = abs(self.alpha)**2 + abs(self.beta)**2
        return abs(norm_sq - 1.0) < tolerance
    
    def __str__(self) -> str:
        return f"({self.alpha}·{self.term0} + {self.beta}·{self.term1})"
    
    def __repr__(self) -> str:
        return f"Superposition({self.alpha!r}, {self.term0!r}, {self.beta!r}, {self.term1!r})"


@dataclass(frozen=True)
class Abstraction(Term):
    """
    Lambda abstraction: λx.M
    
    In linear type system, the bound variable must be used exactly once
    (unless wrapped in a modality).
    """
    var: str
    body: Term
    
    def free_variables(self) -> Set[str]:
        return self.body.free_variables() - {self.var}
    
    def substitute(self, var: str, replacement: Term) -> Term:
        if var == self.var:
            # Variable is shadowed
            return self
        if self.var in replacement.free_variables():
            # Need alpha-conversion to avoid capture
            new_var = self._fresh_var(self.var, replacement.free_variables() | self.body.free_variables())
            new_body = self.body.substitute(self.var, Variable(new_var))
            return Abstraction(new_var, new_body.substitute(var, replacement))
        return Abstraction(self.var, self.body.substitute(var, replacement))
    
    @staticmethod
    def _fresh_var(base: str, avoid: Set[str]) -> str:
        """Generate a fresh variable name."""
        i = 0
        while f"{base}{i}" in avoid:
            i += 1
        return f"{base}{i}"
    
    def is_value(self) -> bool:
        return True
    
    def __str__(self) -> str:
        return f"(λ{self.var}. {self.body})"
    
    def __repr__(self) -> str:
        return f"Abstraction({self.var!r}, {self.body!r})"


@dataclass(frozen=True)
class Application(Term):
    """
    Function application: M N
    
    In call-by-value, N must be a value before reduction.
    """
    func: Term
    arg: Term
    
    def free_variables(self) -> Set[str]:
        return self.func.free_variables() | self.arg.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return Application(
            self.func.substitute(var, replacement),
            self.arg.substitute(var, replacement)
        )
    
    def __str__(self) -> str:
        return f"({self.func} {self.arg})"
    
    def __repr__(self) -> str:
        return f"Application({self.func!r}, {self.arg!r})"


@dataclass(frozen=True)
class TensorPair(Term):
    """
    Tensor product pair: M ⊗ N
    
    Represents simultaneous availability of two quantum resources.
    """
    left: Term
    right: Term
    
    def free_variables(self) -> Set[str]:
        return self.left.free_variables() | self.right.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return TensorPair(
            self.left.substitute(var, replacement),
            self.right.substitute(var, replacement)
        )
    
    def is_value(self) -> bool:
        return self.left.is_value() and self.right.is_value()
    
    def __str__(self) -> str:
        return f"({self.left} ⊗ {self.right})"
    
    def __repr__(self) -> str:
        return f"TensorPair({self.left!r}, {self.right!r})"


@dataclass(frozen=True)
class LetTensor(Term):
    """
    Tensor elimination: let x ⊗ y = M in N
    
    Destructs a tensor pair, binding both components.
    """
    var_left: str
    var_right: str
    tensor_term: Term
    body: Term
    
    def free_variables(self) -> Set[str]:
        body_fv = self.body.free_variables() - {self.var_left, self.var_right}
        return self.tensor_term.free_variables() | body_fv
    
    def substitute(self, var: str, replacement: Term) -> Term:
        if var in {self.var_left, self.var_right}:
            return LetTensor(
                self.var_left,
                self.var_right,
                self.tensor_term.substitute(var, replacement),
                self.body
            )
        return LetTensor(
            self.var_left,
            self.var_right,
            self.tensor_term.substitute(var, replacement),
            self.body.substitute(var, replacement)
        )
    
    def __str__(self) -> str:
        return f"(let {self.var_left} ⊗ {self.var_right} = {self.tensor_term} in {self.body})"
    
    def __repr__(self) -> str:
        return f"LetTensor({self.var_left!r}, {self.var_right!r}, {self.tensor_term!r}, {self.body!r})"


@dataclass(frozen=True)
class BangIntro(Term):
    """
    Bang introduction: !M
    
    Marks a term as duplicable (classical/unrestricted).
    """
    term: Term
    
    def free_variables(self) -> Set[str]:
        return self.term.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return BangIntro(self.term.substitute(var, replacement))
    
    def is_value(self) -> bool:
        return self.term.is_value()
    
    def __str__(self) -> str:
        return f"!{self.term}"
    
    def __repr__(self) -> str:
        return f"BangIntro({self.term!r})"


@dataclass(frozen=True)
class LetBang(Term):
    """
    Bang elimination: let !x = M in N
    
    Destructs a bang-typed term, allowing x to be used multiple times in N.
    """
    var: str
    bang_term: Term
    body: Term
    
    def free_variables(self) -> Set[str]:
        body_fv = self.body.free_variables() - {self.var}
        return self.bang_term.free_variables() | body_fv
    
    def substitute(self, var: str, replacement: Term) -> Term:
        if var == self.var:
            return LetBang(
                self.var,
                self.bang_term.substitute(var, replacement),
                self.body
            )
        return LetBang(
            self.var,
            self.bang_term.substitute(var, replacement),
            self.body.substitute(var, replacement)
        )
    
    def __str__(self) -> str:
        return f"(let !{self.var} = {self.bang_term} in {self.body})"
    
    def __repr__(self) -> str:
        return f"LetBang({self.var!r}, {self.bang_term!r}, {self.body!r})"


@dataclass(frozen=True)
class ParagraphIntro(Term):
    """
    Paragraph introduction: §M
    
    Marks a term with bounded duplication depth (for polynomial time).
    """
    term: Term
    
    def free_variables(self) -> Set[str]:
        return self.term.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return ParagraphIntro(self.term.substitute(var, replacement))
    
    def is_value(self) -> bool:
        return self.term.is_value()
    
    def __str__(self) -> str:
        return f"§{self.term}"
    
    def __repr__(self) -> str:
        return f"ParagraphIntro({self.term!r})"


class UnitaryGate(Enum):
    """Standard unitary quantum gates."""
    H = "H"      # Hadamard
    X = "X"      # Pauli-X (NOT)
    Y = "Y"      # Pauli-Y
    Z = "Z"      # Pauli-Z
    S = "S"      # Phase gate
    T = "T"      # π/8 gate
    CNOT = "CNOT"  # Controlled-NOT
    CZ = "CZ"    # Controlled-Z
    SWAP = "SWAP"  # Swap gate
    
    def arity(self) -> int:
        """Return the number of qubits this gate operates on."""
        if self in {UnitaryGate.CNOT, UnitaryGate.CZ, UnitaryGate.SWAP}:
            return 2
        return 1
    
    def matrix(self) -> List[List[complex]]:
        """Return the unitary matrix representation."""
        sqrt2 = 1 / cmath.sqrt(2)
        if self == UnitaryGate.H:
            return [[sqrt2, sqrt2], [sqrt2, -sqrt2]]
        elif self == UnitaryGate.X:
            return [[0, 1], [1, 0]]
        elif self == UnitaryGate.Y:
            return [[0, -1j], [1j, 0]]
        elif self == UnitaryGate.Z:
            return [[1, 0], [0, -1]]
        elif self == UnitaryGate.S:
            return [[1, 0], [0, 1j]]
        elif self == UnitaryGate.T:
            return [[1, 0], [0, cmath.exp(1j * cmath.pi / 4)]]
        elif self == UnitaryGate.CNOT:
            return [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
        elif self == UnitaryGate.CZ:
            return [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
        elif self == UnitaryGate.SWAP:
            return [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
        raise ValueError(f"Unknown gate: {self}")


@dataclass(frozen=True)
class UnitaryApp(Term):
    """
    Unitary gate application: U[M]
    
    Applies a unitary transformation to a quantum term.
    """
    gate: UnitaryGate
    arg: Term
    
    def free_variables(self) -> Set[str]:
        return self.arg.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return UnitaryApp(self.gate, self.arg.substitute(var, replacement))
    
    def __str__(self) -> str:
        return f"{self.gate.value}[{self.arg}]"
    
    def __repr__(self) -> str:
        return f"UnitaryApp({self.gate!r}, {self.arg!r})"


@dataclass(frozen=True)
class QCtrl(Term):
    """
    Quantum control: qctrl(M, N₀, N₁)
    
    Implements coherent quantum branching:
    - If M = |0⟩, result is N₀
    - If M = |1⟩, result is N₁
    - If M = α|0⟩ + β|1⟩, result is α·N₀ + β·N₁ (coherent superposition)
    
    Requires N₀ ⊥ N₁ (orthogonality) for efficient compilation.
    """
    control: Term
    branch0: Term
    branch1: Term
    
    def free_variables(self) -> Set[str]:
        return (self.control.free_variables() | 
                self.branch0.free_variables() | 
                self.branch1.free_variables())
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return QCtrl(
            self.control.substitute(var, replacement),
            self.branch0.substitute(var, replacement),
            self.branch1.substitute(var, replacement)
        )
    
    def __str__(self) -> str:
        return f"qctrl({self.control}, {self.branch0}, {self.branch1})"
    
    def __repr__(self) -> str:
        return f"QCtrl({self.control!r}, {self.branch0!r}, {self.branch1!r})"


class MeasurementBasis(Enum):
    """Standard measurement bases."""
    COMPUTATIONAL = "Z"  # |0⟩, |1⟩ basis
    HADAMARD = "X"       # |+⟩, |-⟩ basis
    Y_BASIS = "Y"        # |+i⟩, |-i⟩ basis


@dataclass(frozen=True)
class Measurement(Term):
    """
    Quantum measurement: meas_B(M)
    
    Measures a quantum term in the specified basis, collapsing superposition.
    This is a non-deterministic operation.
    """
    basis: MeasurementBasis
    arg: Term
    
    def free_variables(self) -> Set[str]:
        return self.arg.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return Measurement(self.basis, self.arg.substitute(var, replacement))
    
    def __str__(self) -> str:
        return f"meas_{self.basis.value}({self.arg})"
    
    def __repr__(self) -> str:
        return f"Measurement({self.basis!r}, {self.arg!r})"


@dataclass(frozen=True)
class New(Term):
    """
    Fresh qubit allocation: new
    
    Creates a new qubit initialized to |0⟩.
    """
    
    def free_variables(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return self
    
    def is_value(self) -> bool:
        return True
    
    def __str__(self) -> str:
        return "new"
    
    def __repr__(self) -> str:
        return "New()"


@dataclass(frozen=True)
class Unit(Term):
    """Unit value: ()"""
    
    def free_variables(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return self
    
    def is_value(self) -> bool:
        return True
    
    def __str__(self) -> str:
        return "()"
    
    def __repr__(self) -> str:
        return "Unit()"


@dataclass(frozen=True)
class Inl(Term):
    """Left injection: inl(M) for sum types A ⊕ B"""
    term: Term
    
    def free_variables(self) -> Set[str]:
        return self.term.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return Inl(self.term.substitute(var, replacement))
    
    def is_value(self) -> bool:
        return self.term.is_value()
    
    def __str__(self) -> str:
        return f"inl({self.term})"
    
    def __repr__(self) -> str:
        return f"Inl({self.term!r})"


@dataclass(frozen=True)
class Inr(Term):
    """Right injection: inr(M) for sum types A ⊕ B"""
    term: Term
    
    def free_variables(self) -> Set[str]:
        return self.term.free_variables()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return Inr(self.term.substitute(var, replacement))
    
    def is_value(self) -> bool:
        return self.term.is_value()
    
    def __str__(self) -> str:
        return f"inr({self.term})"
    
    def __repr__(self) -> str:
        return f"Inr({self.term!r})"


@dataclass(frozen=True)
class Case(Term):
    """
    Case analysis: case M of inl(x) => N₁ | inr(y) => N₂
    """
    scrutinee: Term
    var_left: str
    branch_left: Term
    var_right: str
    branch_right: Term
    
    def free_variables(self) -> Set[str]:
        left_fv = self.branch_left.free_variables() - {self.var_left}
        right_fv = self.branch_right.free_variables() - {self.var_right}
        return self.scrutinee.free_variables() | left_fv | right_fv
    
    def substitute(self, var: str, replacement: Term) -> Term:
        new_scrutinee = self.scrutinee.substitute(var, replacement)
        new_left = self.branch_left if var == self.var_left else self.branch_left.substitute(var, replacement)
        new_right = self.branch_right if var == self.var_right else self.branch_right.substitute(var, replacement)
        return Case(new_scrutinee, self.var_left, new_left, self.var_right, new_right)
    
    def __str__(self) -> str:
        return f"(case {self.scrutinee} of inl({self.var_left}) => {self.branch_left} | inr({self.var_right}) => {self.branch_right})"
    
    def __repr__(self) -> str:
        return f"Case({self.scrutinee!r}, {self.var_left!r}, {self.branch_left!r}, {self.var_right!r}, {self.branch_right!r})"
