"""
QLLR: Quantum Light Linear Realizability Framework

A typed quantum lambda calculus with coherent control ensuring 
polynomial-time normalization through stratified modal types.

Based on the PhD research proposal:
"Resource-Aware Quantum Lambda Calculi with Coherent Control"

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from qllr.core.syntax import (
    Term,
    Ket0,
    Ket1,
    Superposition,
    Variable,
    Abstraction,
    Application,
    TensorPair,
    LetTensor,
    LetBang,
    UnitaryApp,
    QCtrl,
    Measurement,
    New,
)

from qllr.core.types import (
    Type,
    QubitType,
    LinearArrow,
    TensorProduct,
    SumType,
    BangType,
    ParagraphType,
    SharpType,
    ForallType,
)

# Alias for convenience
Qubit = Ket0

from qllr.typing.typechecker import TypeChecker, TypeCheckError
from qllr.typing.orthogonality import OrthogonalityChecker
from qllr.compilation.circuit_extraction import CircuitExtractor
from qllr.compilation.circuit import QuantumCircuit, Gate
from qllr.analysis.complexity import ComplexityAnalyzer
from qllr.analysis.width import WidthAnalyzer

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core syntax
    "Term",
    "Type",
    "Qubit",
    "Ket0",
    "Ket1",
    "Superposition",
    "Variable",
    "Abstraction",
    "Application",
    "TensorPair",
    "LetTensor",
    "LetBang",
    "UnitaryApp",
    "QCtrl",
    "Measurement",
    "New",
    # Types
    "QubitType",
    "LinearArrow",
    "TensorProduct",
    "SumType",
    "BangType",
    "ParagraphType",
    "SharpType",
    "ForallType",
    # Type checking
    "TypeChecker",
    "TypeCheckError",
    "OrthogonalityChecker",
    # Compilation
    "CircuitExtractor",
    "QuantumCircuit",
    "Gate",
    # Analysis
    "ComplexityAnalyzer",
    "WidthAnalyzer",
]
