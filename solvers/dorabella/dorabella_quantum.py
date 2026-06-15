"""Quantum-hybrid candidate sampler.

The real 87-character problem is too large for current public QPUs as one
monolithic Grover oracle. This module therefore exposes a block sampler: each
block is small enough to encode on available hardware, and the symbolic layer
recombines and validates the results with the global bijection.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Sequence

from .dorabella_constraints import CipherState
from .dorabella_data import CANONICAL_LETTERS, DorabellaPosition


@dataclass(frozen=True)
class QuantumConfig:
    mode: str = "local"
    backend: str = "aer_simulator"
    shots: int = 4096
    max_block_unknowns: int = 8
    grover_iterations: int | None = None
    use_real_ibm_backend: bool = False
    ibm_token_env: str = "IBM_QUANTUM_TOKEN"
    ibm_instance: str | None = None


class QuantumBlockSampler:
    """Prepare candidate assignments for unknown slots.

    Modes:
    - local: classical local sampler that mirrors measured assignments.
    - qiskit-aer: real Qiskit circuit sampled on Aer, then symbolically filtered.
    - ibm-runtime: same block circuit submitted through IBM Runtime when tokens
      and dependencies are configured.

    The current circuit is a quantum block sampler, not a full semantic Grover
    oracle. The extension point for a deeper oracle is `build_grover_circuit`.
    """

    def __init__(self, config: QuantumConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()
        self._ibm_service = None
        self._ibm_backend = None
        self._ibm_sampler = None
        self._ibm_pass_manager = None
        self._ibm_transpiled_circuits: dict[int, object] = {}

    def sample_block(
        self,
        block: Sequence[DorabellaPosition],
        state: CipherState,
        forbidden_signatures: set[str],
    ) -> list[dict[int, str]]:
        if len(block) > self.config.max_block_unknowns:
            raise ValueError("block is too large for configured quantum sampler")
        if self.config.mode == "local":
            return self._local_amplitude_sampler(block, state, forbidden_signatures)
        if self.config.mode == "qiskit-aer":
            return self._qiskit_aer_sampler(block, state, forbidden_signatures)
        if self.config.mode == "ibm-runtime":
            return self._ibm_runtime_sampler(block, state, forbidden_signatures)
        raise ValueError(f"unknown quantum mode: {self.config.mode}")

    def _local_amplitude_sampler(
        self,
        block: Sequence[DorabellaPosition],
        state: CipherState,
        forbidden_signatures: set[str],
    ) -> list[dict[int, str]]:
        results: list[dict[int, str]] = []
        trials = max(self.config.shots, 1)
        for _ in range(trials):
            trial = state.clone()
            assignment: dict[int, str] = {}
            ok = True
            for pos in block:
                domain = sorted(pos.allowed_letters or set(CANONICAL_LETTERS))
                self.rng.shuffle(domain)
                placed = False
                for letter in domain:
                    symbol_options = list(pos.symbol_options)
                    self.rng.shuffle(symbol_options)
                    for sym in symbol_options:
                        t2 = trial.clone()
                        if t2.bind(pos.alphabet_index, sym, letter):
                            trial = t2
                            assignment[pos.global_index] = letter
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    ok = False
                    break
            sig = "".join(assignment.get(pos.global_index, "?") for pos in block)
            if ok and sig not in forbidden_signatures:
                results.append(assignment)
        # Preserve order but remove duplicates.
        seen: set[tuple[tuple[int, str], ...]] = set()
        unique: list[dict[int, str]] = []
        for item in results:
            key = tuple(sorted(item.items()))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _domain_for_position(self, pos: DorabellaPosition) -> list[str]:
        return sorted(pos.allowed_letters or set(CANONICAL_LETTERS))

    def _assignment_from_indices(
        self,
        block: Sequence[DorabellaPosition],
        state: CipherState,
        indices: Sequence[int],
    ) -> dict[int, str] | None:
        trial = state.clone()
        assignment: dict[int, str] = {}
        for pos, raw_idx in zip(block, indices):
            domain = self._domain_for_position(pos)
            if not domain:
                return None
            letter = domain[raw_idx % len(domain)]
            placed = False
            for sym in pos.symbol_options:
                t2 = trial.clone()
                if t2.bind(pos.alphabet_index, sym, letter):
                    trial = t2
                    assignment[pos.global_index] = letter
                    placed = True
                    break
            if not placed:
                return None
        return assignment

    def _dedupe_and_filter(
        self,
        block: Sequence[DorabellaPosition],
        candidates: list[dict[int, str]],
        forbidden_signatures: set[str],
    ) -> list[dict[int, str]]:
        seen: set[tuple[tuple[int, str], ...]] = set()
        unique: list[dict[int, str]] = []
        for item in candidates:
            sig = "".join(item.get(pos.global_index, "?") for pos in block)
            key = tuple(sorted(item.items()))
            if sig in forbidden_signatures or key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _bitstring_to_indices(self, bitstring: str, block_size: int) -> list[int]:
        # Qiskit count keys are classical-bit strings. We chunk from the right so
        # qubit 0 is represented in the first block variable.
        padded = bitstring.replace(" ", "").zfill(block_size * 5)
        chunks = []
        for offset in range(0, block_size * 5, 5):
            start = len(padded) - offset - 5
            end = len(padded) - offset
            chunks.append(int(padded[start:end], 2))
        return chunks

    def _qiskit_aer_sampler(
        self,
        block: Sequence[DorabellaPosition],
        state: CipherState,
        forbidden_signatures: set[str],
    ) -> list[dict[int, str]]:
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install qiskit and qiskit-aer to use --quantum-mode qiskit-aer") from exc

        circuit = self.build_quantum_sampling_circuit(block)
        backend = AerSimulator()
        compiled = transpile(circuit, backend)
        result = backend.run(compiled, shots=self.config.shots).result()
        counts = result.get_counts()
        candidates: list[dict[int, str]] = []
        for bitstring, count in counts.items():
            indices = self._bitstring_to_indices(bitstring, len(block))
            assignment = self._assignment_from_indices(block, state, indices)
            if assignment is not None:
                candidates.extend([assignment] * min(count, 4))
        return self._dedupe_and_filter(block, candidates, forbidden_signatures)

    def _ibm_runtime_sampler(
        self,
        block: Sequence[DorabellaPosition],
        state: CipherState,
        forbidden_signatures: set[str],
    ) -> list[dict[int, str]]:
        backend, sampler = self._ibm_runtime_objects()
        circuit = self._ibm_transpiled_sampling_circuit(block, backend)
        job = sampler.run([circuit], shots=self.config.shots)
        result = job.result()

        # Runtime result containers have changed across Qiskit versions. Keep
        # extraction narrow and fail loudly if the installed version differs.
        try:
            counts = result[0].data.meas.get_counts()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Could not read IBM Runtime counts from this qiskit-ibm-runtime version") from exc

        candidates: list[dict[int, str]] = []
        for bitstring, count in counts.items():
            indices = self._bitstring_to_indices(bitstring, len(block))
            assignment = self._assignment_from_indices(block, state, indices)
            if assignment is not None:
                candidates.extend([assignment] * min(count, 4))
        return self._dedupe_and_filter(block, candidates, forbidden_signatures)

    def _ibm_runtime_objects(self):
        token = os.environ.get(self.config.ibm_token_env)
        if not token:
            raise RuntimeError(f"Set {self.config.ibm_token_env} before using --quantum-mode ibm-runtime")
        if self._ibm_backend is not None and self._ibm_sampler is not None:
            return self._ibm_backend, self._ibm_sampler
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install qiskit-ibm-runtime to use --quantum-mode ibm-runtime") from exc

        service_kwargs = {
            "channel": os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform"),
            "token": token,
        }
        if self.config.ibm_instance:
            service_kwargs["instance"] = self.config.ibm_instance
        self._ibm_service = QiskitRuntimeService(**service_kwargs)
        self._ibm_backend = self._ibm_service.backend(self.config.backend)
        self._ibm_sampler = Sampler(self._ibm_backend)
        return self._ibm_backend, self._ibm_sampler

    def _ibm_transpiled_sampling_circuit(self, block: Sequence[DorabellaPosition], backend):
        block_size = len(block)
        cached = self._ibm_transpiled_circuits.get(block_size)
        if cached is not None:
            return cached
        circuit = self.build_quantum_sampling_circuit(block)
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            if self._ibm_pass_manager is None:
                self._ibm_pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
            circuit = self._ibm_pass_manager.run(circuit)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Could not transpile the sampling circuit for the selected IBM backend") from exc
        self._ibm_transpiled_circuits[block_size] = circuit
        return circuit

    def recommended_grover_iterations(self, search_space_size: int, marked_estimate: int = 1) -> int:
        if self.config.grover_iterations is not None:
            return self.config.grover_iterations
        if search_space_size <= 1:
            return 1
        return max(1, int((math.pi / 4.0) * math.sqrt(search_space_size / max(marked_estimate, 1))))

    def build_grover_circuit(self, block: Sequence[DorabellaPosition]):
        """Extension point for a real Qiskit oracle.

        A production oracle should encode:
        - 5 qubits per unknown letter in the block.
        - equality tests for repeated symbols within the same alphabet.
        - inequality tests for two symbols mapping to the same letter.
        - skeleton masks and orientation alternatives.

        The current project intentionally leaves this unexecuted and explicit,
        because IBM hardware access and qubit budget must be chosen by the user.
        """
        try:
            from qiskit import QuantumCircuit
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install qiskit to build quantum circuits") from exc

        qubits_per_letter = 5
        qc = QuantumCircuit(len(block) * qubits_per_letter, name="dorabella_block_grover")
        qc.h(range(qc.num_qubits))
        return qc

    def build_quantum_sampling_circuit(self, block: Sequence[DorabellaPosition]):
        """Build a measured block-superposition circuit.

        This circuit puts each unknown letter variable into a 5-qubit
        superposition, measures it, and lets the symbolic layer map measurements
        into the allowed letter domains. It is deliberately small enough to run
        on Aer and, for tiny blocks, on IBM Runtime.
        """
        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install qiskit to build quantum circuits") from exc

        qubits_per_letter = 5
        qreg = QuantumRegister(len(block) * qubits_per_letter, "q")
        creg = ClassicalRegister(len(block) * qubits_per_letter, "meas")
        qc = QuantumCircuit(qreg, creg, name="dorabella_block_sampler")
        qc.h(qreg)
        qc.measure(qreg, creg)
        return qc
