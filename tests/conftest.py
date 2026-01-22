"""
Pytest configuration and shared fixtures for QLLR tests.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from qllr.core.syntax import (
    Ket0, Ket1, Variable, Abstraction, Application,
    TensorPair, UnitaryGate, UnitaryApp, QCtrl, New
)
from qllr.core.types import QubitType, LinearArrow, TensorProduct
from qllr.typing import TypeChecker, OrthogonalityChecker


@pytest.fixture
def ket0():
    """Fixture for |0⟩ basis state."""
    return Ket0()


@pytest.fixture
def ket1():
    """Fixture for |1⟩ basis state."""
    return Ket1()


@pytest.fixture
def hadamard_ket0():
    """Fixture for H[|0⟩]."""
    return UnitaryApp(UnitaryGate.H, Ket0())


@pytest.fixture
def bell_pair_init():
    """Fixture for H[|0⟩] ⊗ |0⟩ (initial Bell state before CNOT)."""
    return TensorPair(UnitaryApp(UnitaryGate.H, Ket0()), Ket0())


@pytest.fixture
def identity_function():
    """Fixture for λx. x (identity function)."""
    return Abstraction("x", Variable("x"))


@pytest.fixture
def hadamard_function():
    """Fixture for λx. H[x]."""
    return Abstraction("x", UnitaryApp(UnitaryGate.H, Variable("x")))


@pytest.fixture
def simple_qctrl():
    """Fixture for qctrl(|0⟩, |0⟩, |1⟩)."""
    return QCtrl(Ket0(), Ket0(), Ket1())


@pytest.fixture
def type_checker():
    """Fixture for a basic type checker."""
    return TypeChecker()


@pytest.fixture
def type_checker_with_orth():
    """Fixture for type checker with orthogonality checking."""
    return TypeChecker(orthogonality_checker=OrthogonalityChecker())


@pytest.fixture
def orthogonality_checker():
    """Fixture for orthogonality checker."""
    return OrthogonalityChecker()


@pytest.fixture
def qubit_type():
    """Fixture for qubit type."""
    return QubitType()


@pytest.fixture
def qubit_to_qubit():
    """Fixture for qubit ⊸ qubit type."""
    return LinearArrow(QubitType(), QubitType())


@pytest.fixture
def qubit_pair_type():
    """Fixture for qubit ⊗ qubit type."""
    return TensorProduct(QubitType(), QubitType())
