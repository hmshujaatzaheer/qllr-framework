# QLLR Framework

[![Tests](https://github.com/hmshujaatzaheer/qllr-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/hmshujaatzaheer/qllr-framework/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hmshujaatzaheer/qllr-framework/branch/main/graph/badge.svg)](https://codecov.io/gh/hmshujaatzaheer/qllr-framework)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Quantum Light Linear Realizability Framework** - A typed quantum lambda calculus with coherent control ensuring polynomial-time normalization through stratified modal types.

## Overview

QLLR is a comprehensive implementation of the λ^QLLR framework, a quantum programming language that:

- **Guarantees polynomial-time execution** through light linear logic stratification
- **Supports coherent quantum control** with orthogonality-based compilation
- **Provides sound and complete FBQP characterization** for higher-order quantum programs
- **Enables automatic circuit extraction** with verified complexity bounds

This implementation accompanies the PhD research proposal: *"Resource-Aware Quantum Lambda Calculi with Coherent Control"*.

## Features

### Core Language (λ^QLLR)

- **Quantum Types**: `qubit`, tensor products (`A ⊗ B`), linear functions (`A ⊸ B`)
- **Modal Types**: 
  - `!A` (bang): unlimited duplication
  - `§A` (paragraph): bounded duplication for polynomial time
  - `♯A` (sharp): superposition modality
- **Quantum Operations**: Unitary gates, quantum control (`qctrl`), measurement
- **Linear Type System**: Enforces no-cloning theorem at the type level

### Type Checker

- Linear context management (exactly-once usage)
- Modal depth tracking for complexity bounds
- Orthogonality verification for quantum control branches
- Full type inference for common patterns

### Circuit Extraction

- Compiles λ^QLLR terms to quantum circuits
- Orthogonal merging for efficient quantum control
- Closure conversion for higher-order functions
- OpenQASM 3 output generation

### Complexity Analysis

- **Time Complexity**: Polynomial bounds from modal depth
- **Space Complexity**: Width analysis for qubit requirements
- Verification of polynomial-time fragment membership

## Installation

```bash
# Clone the repository
git clone https://github.com/hmshujaatzaheer/qllr-framework.git
cd qllr-framework

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Creating Quantum Terms

```python
from qllr.core import Ket0, Ket1, UnitaryGate, UnitaryApp, TensorPair, QCtrl

# Basis states
zero = Ket0()
one = Ket1()

# Hadamard gate application
h_zero = UnitaryApp(UnitaryGate.H, zero)

# Tensor product (two qubits)
pair = TensorPair(h_zero, zero)

# Quantum control (coherent branching)
ctrl = QCtrl(h_zero, zero, one)
```

### Type Checking

```python
from qllr.typing import TypeChecker, OrthogonalityChecker

# Basic type checking
tc = TypeChecker()
typ = tc.check(h_zero)
print(f"Type: {typ}")  # qubit

# With orthogonality checking
orth_checker = OrthogonalityChecker()
tc = TypeChecker(orthogonality_checker=orth_checker)
typ = tc.check(QCtrl(Ket0(), Ket0(), Ket1()))
```

### Circuit Extraction

```python
from qllr.compilation import extract_circuit

# Extract quantum circuit
circuit = extract_circuit(h_zero)
print(f"Qubits: {circuit.num_qubits}")
print(f"Gates: {circuit.size()}")
print(f"Depth: {circuit.depth()}")

# Generate OpenQASM
qasm = circuit.to_openqasm()
print(qasm)
```

### Complexity Analysis

```python
from qllr.analysis import ComplexityAnalyzer, WidthAnalyzer

# Time complexity
comp = ComplexityAnalyzer()
bound = comp.analyze_term(h_zero)
print(f"Complexity: {bound}")  # O(n^2)

# Space complexity
width = WidthAnalyzer()
w = width.analyze_term(pair)
print(f"Width: {w.width} qubits")  # 2
```

## Project Structure

```
qllr-framework/
├── src/qllr/
│   ├── __init__.py          # Package exports
│   ├── core/
│   │   ├── syntax.py        # AST definitions
│   │   └── types.py         # Type system
│   ├── typing/
│   │   ├── typechecker.py   # Type checking algorithm
│   │   └── orthogonality.py # Orthogonality checking
│   ├── compilation/
│   │   ├── circuit.py       # Quantum circuit representation
│   │   └── circuit_extraction.py  # Term → Circuit
│   └── analysis/
│       ├── complexity.py    # Time complexity analysis
│       └── width.py         # Space complexity analysis
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Example programs
└── docs/                   # Documentation
```

## Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_core.py

# Run with verbose output
pytest -v --tb=long

# Generate coverage report
pytest --cov=qllr --cov-report=html
open htmlcov/index.html
```

## Theoretical Foundation

### FBQP Characterization (Theorem 4.2)

For a well-typed term Γ; Δ ⊢ M : A in λ^QLLR_poly:
- M normalizes in polynomial time (bounded by modal depth)
- The extracted circuit computes a function in FBQP
- Conversely, any FBQP function is expressible in λ^QLLR_poly

### Orthogonality Checking (Algorithm 3)

For quantum control `qctrl(M, N₀, N₁)`:
1. Find structural differences between N₀ and N₁
2. Verify differences don't depend on bound variables
3. Check each difference is first-order orthogonal

### Width Analysis (Theorem 6.1)

For Γ; Δ ⊢_w M : A^n:
- Executing M requires at most n qubits simultaneously
- Width is additive for tensor products
- Width is preserved by unitary operations

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{zaheer2026qllr,
  author = {H M Shujaat Zaheer},
  title = {QLLR Framework: Resource-Aware Quantum Lambda Calculi with Coherent Control},
  year = {2026},
  url = {https://github.com/hmshujaatzaheer/qllr-framework}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**H M Shujaat Zaheer**
- Email: shujabis@gmail.com
- GitHub: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)

## Acknowledgments

This work is based on the PhD research proposal submitted to Inria MOCQUA team, supervised by Dr. Romain Péchoux. The theoretical foundations draw from:

- Light Linear Logic and DLAL [Girard, Lafont]
- Lambda-S₁ Realizability [Díaz-Caro et al., 2019]
- FOQ/PFOQ Languages [Hainry, Péchoux, Silva, 2025]
- Branch Sequentialization [Hainry, Péchoux, Silva, 2025]
