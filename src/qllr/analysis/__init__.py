"""
Analysis module for λ^QLLR.

Provides complexity and space analysis for terms.
"""

from qllr.analysis.complexity import (
    ComplexityBound,
    ComplexityAnalyzer,
    analyze_complexity,
    verify_polytime,
)
from qllr.analysis.width import (
    WidthBound,
    WidthAnalyzer,
    analyze_width,
    compute_qubit_count,
)

__all__ = [
    "ComplexityBound",
    "ComplexityAnalyzer",
    "analyze_complexity",
    "verify_polytime",
    "WidthBound",
    "WidthAnalyzer",
    "analyze_width",
    "compute_qubit_count",
]
