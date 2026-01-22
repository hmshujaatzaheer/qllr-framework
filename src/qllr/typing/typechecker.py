"""
Type Checker for λ^QLLR

Implements the typing rules from Definition 3.2 of the research proposal:
- Linear context (Δ): variables used exactly once
- Unrestricted context (Γ): variables under ! modality
- Typing judgment: Γ; Δ ⊢ M : A

Key properties:
- Preservation: If Γ; Δ ⊢ M : A and M → M', then Γ; Δ ⊢ M' : A
- Progress: If Γ; Δ ⊢ M : A, then M is a value or M → M'
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple, List
from enum import Enum

from qllr.core.syntax import (
    Term, Variable, Ket0, Ket1, Superposition, Abstraction, Application,
    TensorPair, LetTensor, BangIntro, LetBang, ParagraphIntro,
    UnitaryGate, UnitaryApp, QCtrl, MeasurementBasis, Measurement,
    New, Unit, Inl, Inr, Case
)
from qllr.core.types import (
    Type, QubitType, UnitType, BoolType, TypeVariable,
    LinearArrow, TensorProduct, SumType, BangType, ParagraphType,
    SharpType, ForallType, types_equal
)


class TypeCheckError(Exception):
    """Exception raised when type checking fails."""
    pass


@dataclass
class Context:
    """
    Typing context with two zones:
    - unrestricted: Γ (variables under !)
    - linear: Δ (variables used exactly once)
    """
    unrestricted: Dict[str, Type] = field(default_factory=dict)
    linear: Dict[str, Type] = field(default_factory=dict)
    
    def lookup(self, var: str) -> Tuple[Type, bool]:
        """
        Look up a variable in the context.
        Returns (type, is_linear).
        """
        if var in self.linear:
            return (self.linear[var], True)
        if var in self.unrestricted:
            return (self.unrestricted[var], False)
        raise TypeCheckError(f"Variable '{var}' not in scope")
    
    def add_linear(self, var: str, typ: Type) -> 'Context':
        """Add a variable to the linear context."""
        if var in self.linear or var in self.unrestricted:
            raise TypeCheckError(f"Variable '{var}' already in scope")
        return Context(
            unrestricted=self.unrestricted.copy(),
            linear={**self.linear, var: typ}
        )
    
    def add_unrestricted(self, var: str, typ: Type) -> 'Context':
        """Add a variable to the unrestricted context."""
        if var in self.linear or var in self.unrestricted:
            raise TypeCheckError(f"Variable '{var}' already in scope")
        return Context(
            unrestricted={**self.unrestricted, var: typ},
            linear=self.linear.copy()
        )
    
    def remove_linear(self, var: str) -> 'Context':
        """Remove a variable from the linear context (after use)."""
        if var not in self.linear:
            raise TypeCheckError(f"Variable '{var}' not in linear context")
        new_linear = self.linear.copy()
        del new_linear[var]
        return Context(
            unrestricted=self.unrestricted.copy(),
            linear=new_linear
        )
    
    def split(self, vars1: Set[str]) -> Tuple['Context', 'Context']:
        """
        Split the linear context between two sub-derivations.
        vars1 goes to the first context, the rest to the second.
        """
        linear1 = {v: t for v, t in self.linear.items() if v in vars1}
        linear2 = {v: t for v, t in self.linear.items() if v not in vars1}
        return (
            Context(unrestricted=self.unrestricted.copy(), linear=linear1),
            Context(unrestricted=self.unrestricted.copy(), linear=linear2)
        )
    
    def merge(self, other: 'Context') -> 'Context':
        """Merge two contexts (after splitting)."""
        # Check no overlap in linear parts
        overlap = set(self.linear.keys()) & set(other.linear.keys())
        if overlap:
            raise TypeCheckError(f"Linear variables used multiple times: {overlap}")
        return Context(
            unrestricted={**self.unrestricted, **other.unrestricted},
            linear={**self.linear, **other.linear}
        )
    
    def is_empty_linear(self) -> bool:
        """Check if the linear context is empty."""
        return len(self.linear) == 0
    
    def linear_vars(self) -> Set[str]:
        """Return the set of linear variables."""
        return set(self.linear.keys())
    
    def __str__(self) -> str:
        gamma = ", ".join(f"{v}:{t}" for v, t in self.unrestricted.items())
        delta = ", ".join(f"{v}:{t}" for v, t in self.linear.items())
        return f"Γ=[{gamma}]; Δ=[{delta}]"


class TypeChecker:
    """
    Type checker for λ^QLLR with support for:
    - Linear types (no-cloning)
    - Modal types (!, §, ♯)
    - Quantum control with orthogonality checking
    """
    
    def __init__(self, orthogonality_checker=None):
        """
        Initialize the type checker.
        
        Args:
            orthogonality_checker: Optional checker for orthogonality conditions
        """
        self.orthogonality_checker = orthogonality_checker
    
    def check(self, term: Term, context: Optional[Context] = None) -> Type:
        """
        Type check a term and return its type.
        
        Args:
            term: The term to type check
            context: Optional initial context (empty by default)
            
        Returns:
            The type of the term
            
        Raises:
            TypeCheckError: If the term is not well-typed
        """
        if context is None:
            context = Context()
        
        result_type, final_ctx = self._check(term, context)
        
        # All linear variables must be used
        if not final_ctx.is_empty_linear():
            unused = final_ctx.linear_vars()
            raise TypeCheckError(f"Linear variables not used: {unused}")
        
        return result_type
    
    def _check(self, term: Term, ctx: Context) -> Tuple[Type, Context]:
        """
        Internal type checking with context threading.
        Returns (type, remaining_context).
        """
        if isinstance(term, Variable):
            return self._check_variable(term, ctx)
        elif isinstance(term, Ket0):
            return (QubitType(), ctx)
        elif isinstance(term, Ket1):
            return (QubitType(), ctx)
        elif isinstance(term, Superposition):
            return self._check_superposition(term, ctx)
        elif isinstance(term, Abstraction):
            return self._check_abstraction(term, ctx)
        elif isinstance(term, Application):
            return self._check_application(term, ctx)
        elif isinstance(term, TensorPair):
            return self._check_tensor_pair(term, ctx)
        elif isinstance(term, LetTensor):
            return self._check_let_tensor(term, ctx)
        elif isinstance(term, BangIntro):
            return self._check_bang_intro(term, ctx)
        elif isinstance(term, LetBang):
            return self._check_let_bang(term, ctx)
        elif isinstance(term, ParagraphIntro):
            return self._check_paragraph_intro(term, ctx)
        elif isinstance(term, UnitaryApp):
            return self._check_unitary_app(term, ctx)
        elif isinstance(term, QCtrl):
            return self._check_qctrl(term, ctx)
        elif isinstance(term, Measurement):
            return self._check_measurement(term, ctx)
        elif isinstance(term, New):
            return (QubitType(), ctx)
        elif isinstance(term, Unit):
            return (UnitType(), ctx)
        elif isinstance(term, Inl):
            return self._check_inl(term, ctx)
        elif isinstance(term, Inr):
            return self._check_inr(term, ctx)
        elif isinstance(term, Case):
            return self._check_case(term, ctx)
        else:
            raise TypeCheckError(f"Unknown term type: {type(term)}")
    
    def _check_variable(self, term: Variable, ctx: Context) -> Tuple[Type, Context]:
        """
        Variable rule:
        ─────────────────────
        Γ; x:A ⊢ x : A
        """
        typ, is_linear = ctx.lookup(term.name)
        if is_linear:
            # Remove from linear context (used exactly once)
            return (typ, ctx.remove_linear(term.name))
        else:
            # Unrestricted, can be used again
            return (typ, ctx)
    
    def _check_superposition(self, term: Superposition, ctx: Context) -> Tuple[Type, Context]:
        """
        Superposition rule:
        Γ; Δ ⊢ M : A    Γ; Δ ⊢ N : A
        ──────────────────────────────
        Γ; Δ ⊢ α·M + β·N : ♯A
        
        Note: Both branches must have the same type and use the same linear resources.
        """
        # Both terms should have the same type
        typ0, ctx0 = self._check(term.term0, ctx)
        typ1, ctx1 = self._check(term.term1, ctx)
        
        if not types_equal(typ0, typ1):
            raise TypeCheckError(
                f"Superposition branches have different types: {typ0} vs {typ1}"
            )
        
        # Linear contexts should be equal (same resources used in both branches)
        if ctx0.linear != ctx1.linear:
            raise TypeCheckError(
                "Superposition branches use different linear resources"
            )
        
        return (SharpType(typ0), ctx0)
    
    def _check_abstraction(self, term: Abstraction, ctx: Context) -> Tuple[Type, Context]:
        """
        Abstraction rule:
        Γ; Δ, x:A ⊢ M : B
        ──────────────────────
        Γ; Δ ⊢ λx.M : A ⊸ B
        """
        # We need to infer the type of x from usage
        # For now, require type annotation or inference
        # This is a simplified version - full implementation would need type inference
        
        # Check if the variable is used linearly in the body
        if term.var in term.body.free_variables():
            # Infer type from first use - simplified approach
            # In a full implementation, we'd need bidirectional type checking
            
            # For basic cases, try to infer from context of use
            inferred_type = self._infer_param_type(term.var, term.body)
            if inferred_type is None:
                inferred_type = QubitType()  # Default for quantum
            
            extended_ctx = ctx.add_linear(term.var, inferred_type)
            body_type, final_ctx = self._check(term.body, extended_ctx)
            
            # Variable should have been consumed
            if term.var in final_ctx.linear:
                raise TypeCheckError(f"Linear variable '{term.var}' not used in abstraction body")
            
            return (LinearArrow(inferred_type, body_type), final_ctx)
        else:
            # Variable not used - this is only valid if we have weakening (for unit types)
            raise TypeCheckError(f"Linear variable '{term.var}' not used in abstraction body")
    
    def _infer_param_type(self, var: str, body: Term) -> Optional[Type]:
        """
        Attempt to infer the type of a parameter from its usage.
        This is a simplified heuristic.
        """
        # Check for unitary application
        if isinstance(body, UnitaryApp) and isinstance(body.arg, Variable) and body.arg.name == var:
            return QubitType()
        
        # Check for application
        if isinstance(body, Application):
            if isinstance(body.arg, Variable) and body.arg.name == var:
                # Could be any type - need more context
                return None
        
        # Check in subterms
        if isinstance(body, Application):
            result = self._infer_param_type(var, body.func)
            if result:
                return result
            return self._infer_param_type(var, body.arg)
        
        if isinstance(body, UnitaryApp):
            return self._infer_param_type(var, body.arg)
        
        if isinstance(body, TensorPair):
            result = self._infer_param_type(var, body.left)
            if result:
                return result
            return self._infer_param_type(var, body.right)
        
        return None
    
    def _check_application(self, term: Application, ctx: Context) -> Tuple[Type, Context]:
        """
        Application rule:
        Γ; Δ₁ ⊢ M : A ⊸ B    Γ; Δ₂ ⊢ N : A
        ─────────────────────────────────────
        Γ; Δ₁, Δ₂ ⊢ M N : B
        """
        # Split context based on free variables
        func_vars = term.func.free_variables() & ctx.linear_vars()
        ctx_func, ctx_arg = ctx.split(func_vars)
        
        func_type, ctx_func_after = self._check(term.func, ctx_func)
        
        if not isinstance(func_type, LinearArrow):
            raise TypeCheckError(f"Expected function type, got {func_type}")
        
        arg_type, ctx_arg_after = self._check(term.arg, ctx_arg)
        
        if not types_equal(func_type.domain, arg_type):
            raise TypeCheckError(
                f"Type mismatch: expected {func_type.domain}, got {arg_type}"
            )
        
        final_ctx = ctx_func_after.merge(ctx_arg_after)
        return (func_type.codomain, final_ctx)
    
    def _check_tensor_pair(self, term: TensorPair, ctx: Context) -> Tuple[Type, Context]:
        """
        Tensor introduction:
        Γ; Δ₁ ⊢ M : A    Γ; Δ₂ ⊢ N : B
        ────────────────────────────────
        Γ; Δ₁, Δ₂ ⊢ M ⊗ N : A ⊗ B
        """
        left_vars = term.left.free_variables() & ctx.linear_vars()
        ctx_left, ctx_right = ctx.split(left_vars)
        
        left_type, ctx_left_after = self._check(term.left, ctx_left)
        right_type, ctx_right_after = self._check(term.right, ctx_right)
        
        final_ctx = ctx_left_after.merge(ctx_right_after)
        return (TensorProduct(left_type, right_type), final_ctx)
    
    def _check_let_tensor(self, term: LetTensor, ctx: Context) -> Tuple[Type, Context]:
        """
        Tensor elimination:
        Γ; Δ₁ ⊢ M : A ⊗ B    Γ; Δ₂, x:A, y:B ⊢ N : C
        ───────────────────────────────────────────────
        Γ; Δ₁, Δ₂ ⊢ let x ⊗ y = M in N : C
        """
        tensor_vars = term.tensor_term.free_variables() & ctx.linear_vars()
        ctx_tensor, ctx_body = ctx.split(tensor_vars)
        
        tensor_type, ctx_tensor_after = self._check(term.tensor_term, ctx_tensor)
        
        if not isinstance(tensor_type, TensorProduct):
            raise TypeCheckError(f"Expected tensor type, got {tensor_type}")
        
        # Add bound variables to context
        ctx_extended = ctx_body.add_linear(term.var_left, tensor_type.left)
        ctx_extended = ctx_extended.add_linear(term.var_right, tensor_type.right)
        
        body_type, ctx_body_after = self._check(term.body, ctx_extended)
        
        final_ctx = ctx_tensor_after.merge(ctx_body_after)
        return (body_type, final_ctx)
    
    def _check_bang_intro(self, term: BangIntro, ctx: Context) -> Tuple[Type, Context]:
        """
        Bang introduction:
        Γ; · ⊢ M : A
        ─────────────────
        Γ; · ⊢ !M : !A
        
        Note: Linear context must be empty.
        """
        if not ctx.is_empty_linear():
            raise TypeCheckError("Cannot introduce ! with non-empty linear context")
        
        inner_type, final_ctx = self._check(term.term, ctx)
        return (BangType(inner_type), final_ctx)
    
    def _check_let_bang(self, term: LetBang, ctx: Context) -> Tuple[Type, Context]:
        """
        Bang elimination:
        Γ; Δ₁ ⊢ M : !A    Γ, x:A; Δ₂ ⊢ N : B
        ─────────────────────────────────────
        Γ; Δ₁, Δ₂ ⊢ let !x = M in N : B
        """
        bang_vars = term.bang_term.free_variables() & ctx.linear_vars()
        ctx_bang, ctx_body = ctx.split(bang_vars)
        
        bang_type, ctx_bang_after = self._check(term.bang_term, ctx_bang)
        
        if not isinstance(bang_type, BangType):
            raise TypeCheckError(f"Expected bang type, got {bang_type}")
        
        # Add to unrestricted context (can be used multiple times)
        ctx_extended = ctx_body.add_unrestricted(term.var, bang_type.inner)
        
        body_type, ctx_body_after = self._check(term.body, ctx_extended)
        
        final_ctx = ctx_bang_after.merge(ctx_body_after)
        return (body_type, final_ctx)
    
    def _check_paragraph_intro(self, term: ParagraphIntro, ctx: Context) -> Tuple[Type, Context]:
        """
        Paragraph introduction:
        Γ; §Δ ⊢ M : A
        ─────────────────
        Γ; §Δ ⊢ §M : §A
        
        The paragraph modality requires all linear variables to also be
        under §. This is checked through context transformation.
        """
        # In full DLAL, we'd transform the context
        # Simplified: just check the inner term
        inner_type, final_ctx = self._check(term.term, ctx)
        return (ParagraphType(inner_type), final_ctx)
    
    def _check_unitary_app(self, term: UnitaryApp, ctx: Context) -> Tuple[Type, Context]:
        """
        Unitary application:
        Γ; Δ ⊢ M : qubit^n
        ───────────────────────
        Γ; Δ ⊢ U[M] : qubit^n
        
        where n is the arity of U.
        """
        arg_type, final_ctx = self._check(term.arg, ctx)
        
        expected_arity = term.gate.arity()
        
        if expected_arity == 1:
            if not isinstance(arg_type, QubitType):
                raise TypeCheckError(f"Unitary {term.gate.value} expects qubit, got {arg_type}")
            return (QubitType(), final_ctx)
        else:
            # Multi-qubit gate
            if not self._is_qubit_tensor(arg_type, expected_arity):
                raise TypeCheckError(
                    f"Unitary {term.gate.value} expects {expected_arity} qubits, got {arg_type}"
                )
            return (arg_type, final_ctx)
    
    def _is_qubit_tensor(self, typ: Type, n: int) -> bool:
        """Check if type is a tensor of n qubits."""
        if n == 1:
            return isinstance(typ, QubitType)
        if isinstance(typ, TensorProduct):
            return self._is_qubit_tensor(typ.left, 1) and self._is_qubit_tensor(typ.right, n - 1)
        return False
    
    def _check_qctrl(self, term: QCtrl, ctx: Context) -> Tuple[Type, Context]:
        """
        Quantum control rule:
        Γ; Δ₁ ⊢ M : ♯qubit    Γ; Δ₂ ⊢ N₀ : A    Γ; Δ₂ ⊢ N₁ : A    N₀ ⊥ N₁
        ─────────────────────────────────────────────────────────────────────
        Γ; Δ₁, Δ₂ ⊢ qctrl(M, N₀, N₁) : ♯A
        
        Requires orthogonality of branches for efficient compilation.
        """
        ctrl_vars = term.control.free_variables() & ctx.linear_vars()
        ctx_ctrl, ctx_branches = ctx.split(ctrl_vars)
        
        ctrl_type, ctx_ctrl_after = self._check(term.control, ctx_ctrl)
        
        # Control must be a (possibly superposed) qubit
        if isinstance(ctrl_type, SharpType):
            if not isinstance(ctrl_type.inner, QubitType):
                raise TypeCheckError(f"qctrl control must be qubit, got {ctrl_type}")
        elif not isinstance(ctrl_type, QubitType):
            raise TypeCheckError(f"qctrl control must be qubit, got {ctrl_type}")
        
        # Both branches use the same resources
        branch0_type, _ = self._check(term.branch0, ctx_branches)
        branch1_type, ctx_branches_after = self._check(term.branch1, ctx_branches)
        
        if not types_equal(branch0_type, branch1_type):
            raise TypeCheckError(
                f"qctrl branches have different types: {branch0_type} vs {branch1_type}"
            )
        
        # Check orthogonality if checker is provided
        if self.orthogonality_checker is not None:
            if not self.orthogonality_checker.check_orthogonal(term.branch0, term.branch1):
                raise TypeCheckError("qctrl branches are not orthogonal")
        
        final_ctx = ctx_ctrl_after.merge(ctx_branches_after)
        return (SharpType(branch0_type), final_ctx)
    
    def _check_measurement(self, term: Measurement, ctx: Context) -> Tuple[Type, Context]:
        """
        Measurement rule:
        Γ; Δ ⊢ M : qubit
        ──────────────────────
        Γ; Δ ⊢ meas_B(M) : bool
        """
        arg_type, final_ctx = self._check(term.arg, ctx)
        
        # Can measure qubit or ♯qubit
        if isinstance(arg_type, SharpType):
            if not isinstance(arg_type.inner, QubitType):
                raise TypeCheckError(f"Cannot measure non-qubit type: {arg_type}")
        elif not isinstance(arg_type, QubitType):
            raise TypeCheckError(f"Cannot measure non-qubit type: {arg_type}")
        
        return (BoolType(), final_ctx)
    
    def _check_inl(self, term: Inl, ctx: Context) -> Tuple[Type, Context]:
        """
        Left injection rule - requires type annotation for full type.
        For now, return a placeholder sum type.
        """
        inner_type, final_ctx = self._check(term.term, ctx)
        # In a full implementation, we'd need the right type from context
        return (SumType(inner_type, UnitType()), final_ctx)
    
    def _check_inr(self, term: Inr, ctx: Context) -> Tuple[Type, Context]:
        """
        Right injection rule - requires type annotation for full type.
        """
        inner_type, final_ctx = self._check(term.term, ctx)
        return (SumType(UnitType(), inner_type), final_ctx)
    
    def _check_case(self, term: Case, ctx: Context) -> Tuple[Type, Context]:
        """
        Case analysis rule.
        """
        scrut_vars = term.scrutinee.free_variables() & ctx.linear_vars()
        ctx_scrut, ctx_branches = ctx.split(scrut_vars)
        
        scrut_type, ctx_scrut_after = self._check(term.scrutinee, ctx_scrut)
        
        if not isinstance(scrut_type, SumType):
            raise TypeCheckError(f"Expected sum type, got {scrut_type}")
        
        # Check left branch
        ctx_left = ctx_branches.add_linear(term.var_left, scrut_type.left)
        left_type, _ = self._check(term.branch_left, ctx_left)
        
        # Check right branch
        ctx_right = ctx_branches.add_linear(term.var_right, scrut_type.right)
        right_type, ctx_branches_after = self._check(term.branch_right, ctx_right)
        
        if not types_equal(left_type, right_type):
            raise TypeCheckError(
                f"Case branches have different types: {left_type} vs {right_type}"
            )
        
        final_ctx = ctx_scrut_after.merge(ctx_branches_after)
        return (left_type, final_ctx)
