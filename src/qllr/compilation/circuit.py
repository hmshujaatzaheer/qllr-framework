"""
Quantum Circuit Representation

Defines the target representation for circuit extraction from λ^QLLR terms.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
import cmath


class GateType(Enum):
    """Standard quantum gate types."""
    # Single-qubit gates
    I = "I"       # Identity
    H = "H"       # Hadamard
    X = "X"       # Pauli-X (NOT)
    Y = "Y"       # Pauli-Y
    Z = "Z"       # Pauli-Z
    S = "S"       # Phase gate (√Z)
    S_DAG = "S†"  # S-dagger
    T = "T"       # T gate (π/8)
    T_DAG = "T†"  # T-dagger
    
    # Two-qubit gates
    CNOT = "CNOT"   # Controlled-NOT
    CZ = "CZ"       # Controlled-Z
    SWAP = "SWAP"   # Swap
    
    # Controlled versions
    CTRL = "CTRL"   # Generic controlled gate


@dataclass
class Gate:
    """
    A quantum gate in a circuit.
    
    Attributes:
        gate_type: The type of gate
        target_qubits: Indices of target qubits
        control_qubits: Indices of control qubits (for controlled gates)
        parameters: Optional parameters (for rotation gates)
    """
    gate_type: GateType
    target_qubits: Tuple[int, ...]
    control_qubits: Tuple[int, ...] = field(default_factory=tuple)
    parameters: Tuple[float, ...] = field(default_factory=tuple)
    
    def arity(self) -> int:
        """Total number of qubits this gate operates on."""
        return len(self.target_qubits) + len(self.control_qubits)
    
    def all_qubits(self) -> Set[int]:
        """Return all qubits this gate touches."""
        return set(self.target_qubits) | set(self.control_qubits)
    
    def is_controlled(self) -> bool:
        """Check if this is a controlled gate."""
        return len(self.control_qubits) > 0
    
    def __str__(self) -> str:
        if self.control_qubits:
            ctrl_str = ",".join(map(str, self.control_qubits))
            tgt_str = ",".join(map(str, self.target_qubits))
            return f"C[{ctrl_str}]-{self.gate_type.value}[{tgt_str}]"
        else:
            tgt_str = ",".join(map(str, self.target_qubits))
            return f"{self.gate_type.value}[{tgt_str}]"


@dataclass
class QuantumCircuit:
    """
    A quantum circuit as a sequence of gates.
    
    This is the target representation for compilation from λ^QLLR.
    
    Attributes:
        num_qubits: Total number of qubits
        gates: List of gates in order of application
        input_qubits: Indices of input qubits
        output_qubits: Indices of output qubits
        ancilla_qubits: Indices of ancilla qubits
    """
    num_qubits: int
    gates: List[Gate] = field(default_factory=list)
    input_qubits: Tuple[int, ...] = field(default_factory=tuple)
    output_qubits: Tuple[int, ...] = field(default_factory=tuple)
    ancilla_qubits: Tuple[int, ...] = field(default_factory=tuple)
    
    def add_gate(self, gate: Gate) -> None:
        """Add a gate to the circuit."""
        # Validate qubit indices
        for q in gate.all_qubits():
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} out of range [0, {self.num_qubits})")
        self.gates.append(gate)
    
    def add_h(self, qubit: int) -> None:
        """Add a Hadamard gate."""
        self.add_gate(Gate(GateType.H, (qubit,)))
    
    def add_x(self, qubit: int) -> None:
        """Add a Pauli-X gate."""
        self.add_gate(Gate(GateType.X, (qubit,)))
    
    def add_y(self, qubit: int) -> None:
        """Add a Pauli-Y gate."""
        self.add_gate(Gate(GateType.Y, (qubit,)))
    
    def add_z(self, qubit: int) -> None:
        """Add a Pauli-Z gate."""
        self.add_gate(Gate(GateType.Z, (qubit,)))
    
    def add_s(self, qubit: int) -> None:
        """Add an S gate."""
        self.add_gate(Gate(GateType.S, (qubit,)))
    
    def add_t(self, qubit: int) -> None:
        """Add a T gate."""
        self.add_gate(Gate(GateType.T, (qubit,)))
    
    def add_cnot(self, control: int, target: int) -> None:
        """Add a CNOT gate."""
        self.add_gate(Gate(GateType.CNOT, (target,), (control,)))
    
    def add_cz(self, control: int, target: int) -> None:
        """Add a CZ gate."""
        self.add_gate(Gate(GateType.CZ, (target,), (control,)))
    
    def add_swap(self, qubit1: int, qubit2: int) -> None:
        """Add a SWAP gate."""
        self.add_gate(Gate(GateType.SWAP, (qubit1, qubit2)))
    
    def add_controlled(
        self, 
        gate_type: GateType, 
        control: int, 
        target: int
    ) -> None:
        """Add a controlled version of a single-qubit gate."""
        self.add_gate(Gate(gate_type, (target,), (control,)))
    
    def depth(self) -> int:
        """
        Calculate the circuit depth.
        
        Depth is the maximum number of time steps needed,
        where gates on disjoint qubits can execute in parallel.
        """
        if not self.gates:
            return 0
        
        # Track when each qubit is next available
        qubit_time: Dict[int, int] = {i: 0 for i in range(self.num_qubits)}
        
        for gate in self.gates:
            qubits = gate.all_qubits()
            # Gate executes at max time of all involved qubits
            start_time = max(qubit_time.get(q, 0) for q in qubits)
            # Update availability
            for q in qubits:
                qubit_time[q] = start_time + 1
        
        return max(qubit_time.values()) if qubit_time else 0
    
    def size(self) -> int:
        """Return the number of gates in the circuit."""
        return len(self.gates)
    
    def width(self) -> int:
        """Return the number of qubits."""
        return self.num_qubits
    
    def t_count(self) -> int:
        """Count the number of T gates (for fault-tolerant cost estimation)."""
        return sum(1 for g in self.gates if g.gate_type in {GateType.T, GateType.T_DAG})
    
    def compose(self, other: 'QuantumCircuit') -> 'QuantumCircuit':
        """
        Compose two circuits sequentially.
        
        The output of self becomes the input of other.
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError("Cannot compose circuits with different qubit counts")
        
        result = QuantumCircuit(
            num_qubits=self.num_qubits,
            input_qubits=self.input_qubits,
            output_qubits=other.output_qubits,
        )
        result.gates = self.gates.copy() + other.gates.copy()
        return result
    
    def tensor(self, other: 'QuantumCircuit') -> 'QuantumCircuit':
        """
        Tensor product of two circuits (parallel composition).
        """
        offset = self.num_qubits
        result = QuantumCircuit(
            num_qubits=self.num_qubits + other.num_qubits,
            input_qubits=self.input_qubits + tuple(q + offset for q in other.input_qubits),
            output_qubits=self.output_qubits + tuple(q + offset for q in other.output_qubits),
        )
        
        # Copy gates from self
        result.gates = self.gates.copy()
        
        # Copy gates from other with shifted indices
        for gate in other.gates:
            shifted_targets = tuple(q + offset for q in gate.target_qubits)
            shifted_controls = tuple(q + offset for q in gate.control_qubits)
            result.gates.append(Gate(
                gate.gate_type,
                shifted_targets,
                shifted_controls,
                gate.parameters
            ))
        
        return result
    
    def inverse(self) -> 'QuantumCircuit':
        """
        Return the inverse (adjoint) circuit.
        
        For unitary circuits, this reverses the gate order and
        takes the adjoint of each gate.
        """
        result = QuantumCircuit(
            num_qubits=self.num_qubits,
            input_qubits=self.output_qubits,
            output_qubits=self.input_qubits,
        )
        
        # Adjoint map for gates
        adjoint_map = {
            GateType.S: GateType.S_DAG,
            GateType.S_DAG: GateType.S,
            GateType.T: GateType.T_DAG,
            GateType.T_DAG: GateType.T,
        }
        
        for gate in reversed(self.gates):
            adj_type = adjoint_map.get(gate.gate_type, gate.gate_type)
            result.gates.append(Gate(
                adj_type,
                gate.target_qubits,
                gate.control_qubits,
                gate.parameters
            ))
        
        return result
    
    def to_openqasm(self) -> str:
        """
        Convert to OpenQASM 3 format.
        """
        lines = [
            "OPENQASM 3;",
            'include "stdgates.inc";',
            f"qubit[{self.num_qubits}] q;",
            ""
        ]
        
        gate_map = {
            GateType.I: "id",
            GateType.H: "h",
            GateType.X: "x",
            GateType.Y: "y",
            GateType.Z: "z",
            GateType.S: "s",
            GateType.S_DAG: "sdg",
            GateType.T: "t",
            GateType.T_DAG: "tdg",
            GateType.CNOT: "cx",
            GateType.CZ: "cz",
            GateType.SWAP: "swap",
        }
        
        for gate in self.gates:
            name = gate_map.get(gate.gate_type, str(gate.gate_type.value).lower())
            
            if gate.control_qubits:
                if gate.gate_type == GateType.CNOT:
                    lines.append(f"cx q[{gate.control_qubits[0]}], q[{gate.target_qubits[0]}];")
                elif gate.gate_type == GateType.CZ:
                    lines.append(f"cz q[{gate.control_qubits[0]}], q[{gate.target_qubits[0]}];")
                else:
                    # Generic controlled gate
                    ctrl = gate.control_qubits[0]
                    tgt = gate.target_qubits[0]
                    lines.append(f"ctrl @ {name} q[{ctrl}], q[{tgt}];")
            else:
                qubits = ", ".join(f"q[{q}]" for q in gate.target_qubits)
                lines.append(f"{name} {qubits};")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        header = f"Circuit({self.num_qubits} qubits, {self.size()} gates, depth {self.depth()})"
        gate_strs = [str(g) for g in self.gates]
        return header + "\n  " + "\n  ".join(gate_strs) if gate_strs else header


def identity_circuit(num_qubits: int) -> QuantumCircuit:
    """Create an identity circuit (no gates)."""
    return QuantumCircuit(
        num_qubits=num_qubits,
        input_qubits=tuple(range(num_qubits)),
        output_qubits=tuple(range(num_qubits))
    )


def bell_state_circuit() -> QuantumCircuit:
    """Create a circuit that prepares a Bell state |Φ⁺⟩."""
    circuit = QuantumCircuit(
        num_qubits=2,
        input_qubits=(0, 1),
        output_qubits=(0, 1)
    )
    circuit.add_h(0)
    circuit.add_cnot(0, 1)
    return circuit
